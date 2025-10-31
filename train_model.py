import os
import warnings
import logging
import sys
import traceback
import random
import torch
import click
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import Dataset
from datasets import concatenate_datasets
from torch import nn
from transformers import EarlyStoppingCallback
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


@click.command()
@click.argument('data_path')
@click.argument('output_dir')

def main(data_path, output_dir):
    """Main function to estimate a Trait2Vec model.
    """
    #PARAMETERS
    
    output_dimension = 256
    max_seq_length = 256
    frozen = False
    BF16 = False

    seed = 42 
    n_train_datapoints = 50000
    n_val_datapoints = 3000

    learning_rate = 3.088725904602452e-05
    warmup_ratio = 0.018257384689084222
    early_stopping_patience = 5
    num_train_iters = 20
    train_batch_size = 100 #64
    num_epochs = 10
    eval_steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} loaded")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    word_embedding_model = models.Transformer('sentence-transformers/all-mpnet-base-v2',
                                              max_seq_length=max_seq_length)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_cls_token=True,
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_max_tokens=False,
                                       pooling_mode_weightedmean_tokens = False,
                                       pooling_mode_lasttoken = False)

    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                               out_features=output_dimension,
                               activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model,
                                        dense_model]).to(device)


    #Freeze LLM parameters
    if frozen:
        for param in model[0].parameters():
            param.requires_grad = False

    # 2. Load the Full KB-dataset and transform to STSB format
    df_ip = pd.read_csv(data_path, compression='gzip', sep="\t")
    df_ip['simGIC'] = df_ip['simGIC']/100.0 # normalize similarity score values (per the max SimGIC value)
    df_ip['simGIC'] = (df_ip['simGIC']*2)-1  # to match cos_similarity range
    print("IP file read")

    # Get unique character IDs
    ch1_ids = df_ip['character_1'].unique()
    ch2_ids = df_ip['character_2'].unique()
    ch_ids = np.union1d(ch1_ids, ch2_ids)
    num_ch_ids = len(ch_ids)
    print(f"There are {num_ch_ids} unique character ids")

    num_train_characters = int(len(ch_ids) * 0.5) # this fraction makes the final percentage of comparisons close to 0.7

    # Randomly select the training and test character IDs
    train_character_ids = np.random.choice(ch_ids, num_train_characters, replace=False)
    dev_tst_character_ids = np.setdiff1d(ch_ids, train_character_ids)
    np.random.shuffle(dev_tst_character_ids)
    # Split into two equal sets
    mid_point = len(dev_tst_character_ids) // 2
    dev_character_ids = dev_tst_character_ids[:mid_point]
    tst_character_ids = dev_tst_character_ids[mid_point:]

    train_df = df_ip[df_ip['character_1'].isin(train_character_ids) & df_ip['character_2'].isin(train_character_ids)]
    val_df = df_ip[df_ip['character_1'].isin(dev_character_ids) & df_ip['character_2'].isin(dev_character_ids)]
    tst_df = df_ip[df_ip['character_1'].isin(tst_character_ids) & df_ip['character_2'].isin(tst_character_ids)]

    train = train_df[['desc_1', 'desc_2', 'simGIC']]
    train.columns = ['sentence1', 'sentence2', 'score']
    val = val_df[['desc_1', 'desc_2', 'simGIC']]
    val.columns = ['sentence1', 'sentence2', 'score']
    test = tst_df[['desc_1', 'desc_2', 'simGIC']]
    test.columns = ['sentence1', 'sentence2', 'score']

    print("Train set created: ", train.shape)
    print("Eval set created: ", val.shape)
    print("Test set created: ", test.shape)


    # Drop index columns
    train.reset_index(drop = True, inplace = True)
    val.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)

    # cast sentence1, sentence2 column types to string
    train['sentence1'] = train['sentence1'].astype("string")
    train['sentence2'] = train['sentence2'].astype("string")
    val['sentence1'] = val['sentence1'].astype("string")
    val['sentence2'] = val['sentence2'].astype("string")
    test['sentence1'] = test['sentence1'].astype("string")
    test['sentence2'] = test['sentence2'].astype("string")

    # In[122]:
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate = learning_rate,
        warmup_ratio = warmup_ratio,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=BF16,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=eval_steps, #2000
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=1,
        greater_is_better=False,
        load_best_model_at_end = True,
    )

    early_stopper = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience, # you can change this value if needed
        early_stopping_threshold=0.01 # you can change this value if needed
    )

    ds_test = Dataset.from_pandas(test)
    ds_train = Dataset.from_pandas(train)
    ds_val = Dataset.from_pandas(val)

    for i in range(num_train_iters):
        print(f"Starting train iteration: {i}")
        # Convert to STSB format using Datasets class
        trainer = SentenceTransformerTrainer(
            model = model,
            args = args,
            train_dataset = ds_train.shuffle(seed+i).select(range(n_train_datapoints)),
            eval_dataset = ds_val.shuffle(seed+i).select(range(n_val_datapoints)),
            compute_metrics = None,
            loss = losses.CoSENTLoss(model),
            seed=seed+i,
            callbacks=[early_stopper],
        )
        print("Starting training")
        trainer.train()

        del trainer

    print("Test Evaluator Started...")
    # 7. Evaluate the model performance on the Phenoscape test dataset
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=ds_test["sentence1"], #test["sentence1"]
        sentences2=ds_test["sentence2"], #test["sentence2"]
        scores=ds_test["score"], #test["score"]
        show_progress_bar = True,
        write_csv=True,
        name="pheno-test",
    )
    test_history = test_evaluator(model)
    print("Finetuned model tested")
    print(test_history)

    # 8. Save the trained & evaluated model locally
    final_output_dir = os.path.join(output_dir, "model")
    model.save(final_output_dir)

if __name__ == "__main__":
    main()

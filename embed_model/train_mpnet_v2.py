import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import logging
import sys
import traceback
from datetime import datetime

from datasets import Dataset

import torch
from torch import nn
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

torch.cuda.empty_cache()
#torch.cuda.set_device(args.local_rank)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device loaded")
#print(f'There are {torch.cuda.device_count()} GPU(s) available.')


# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

nm = 'all-mpnet-base-v2'
model_dir = "/projects/imageomics/skar/ht_10p/model_10p/"
data_dir = "/projects/imageomics/skar/ht_10p/data/"
output_dir = (model_dir + nm + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Specify any pre-trained Sentence Transformer model here, e.g., bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'sentence-transformers/'+ nm

# 1. Define the SentenceTransformer model.
output_dimension = 256
max_seq_length = 256
train_batch_size = 64
eval_batch_size = 128
num_epochs = 2


word_embedding_model = models.Transformer(model_name,
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

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
print(nm, ": Pretrained model loaded with custom sequence and embedding lengths")

# 2. Load the Full KB-dataset and transform to STSB format
df_ip = pd.read_csv(data_dir + 'id12_desc12_simGIC.tsv.gz', compression='gzip', sep="\t")
#df_ip = pd.read_pickle(data_dir +'id12_desc12_simGIC_filt.pkl', compression='infer')
print("IP file read")

# create data partitions
df = df_ip[['desc_1', 'desc_2', 'simGIC_1']]
df.columns = ['sentence1', 'sentence2', 'score']
df['score'] = df['score']/100.0 # normalize similarity score values (per the max SimGIC value)

def sample_split(df, perc, tst_split):
    # Shuffle and split the DataFrame
    df_trn = df.sample(frac=perc, random_state=42)
    remaining_df = df.drop(df_trn.index)

    df_dev = remaining_df.sample(frac=tst_split, random_state=42)
    df_tst = remaining_df.drop(df_dev.index).sample(frac=1, random_state=42)

    return df_trn, df_dev, df_tst

train, val, test = sample_split(df=df, perc=0.7, tst_split=0.5)


print("Train set created: ", train.shape)
print("Eval set created: ", val.shape)
print("Test set created: ", test.shape)

#logging.info(train_dataset)

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

# Convert to STSB format using Datasets class
ds_train = Dataset.from_pandas(train)
ds_val = Dataset.from_pandas(val)
ds_test = Dataset.from_pandas(test)
print("Data partitions transformed to STSB format")


train_loss = losses.CosineSimilarityLoss(model=model)
# train_loss = losses.CoSENTLoss(model=model)
# train_loss_angle = losses.AnglELoss(model=model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val["sentence1"],
    sentences2=val["sentence2"],
    scores=val["score"],
    show_progress_bar = True,
 #   main_similarity=SimilarityFunction.COSINE,
    write_csv=True,
    name="pheno-dev",
)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    auto_find_batch_size = True,
    learning_rate = 2e-05,
    #warmup_ratio=0.000001,
    warmup_steps=500,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=50000,
    save_strategy="steps",
    save_steps=50000,
    save_total_limit=5,
    logging_steps=50000,
)

# 6. Create an evaluator & evaluate the base model
trainer = SentenceTransformerTrainer(
    model = model,
    args = args,
    train_dataset = ds_train,
    eval_dataset = ds_val,
    loss = train_loss,
    evaluator = dev_evaluator,
)
print("Trainer Defined...")
trainer.train()

print("Test Evaluator Started...")
# 7. Evaluate the model performance on the Phenoscape test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test["sentence1"],
    sentences2=test["sentence2"],
    scores=test["score"],
    show_progress_bar = True,
 #  main_similarity=SimilarityFunction.COSINE,
    write_csv=True,
    name="pheno-test",
)
test_evaluator(model)
print("Finetuned model tested")

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)


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

#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch
from torch import nn
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

data_file = sys.argv[1] #gzip-compressed TSV
output_dir = sys.argv[2]

torch.cuda.empty_cache()
#torch.cuda.set_device(args.local_rank)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device loaded")
#print(f'There are {torch.cuda.device_count()} GPU(s) available.')


# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

nm = 'all-mpnet-base-v2'
#model_dir = "/projects/imageomics/skar/ht_filtered/model_filt/"
#data_dir = "/projects/imageomics/skar/ht_filtered/data/"
#output_dir = (model_dir + nm + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Specify any Hugging Face pre-trained model here, e.g., bert-base-uncased, roberta-base, xlm-roberta-base
# nm = 'all-mpnet-base-v2'
model_name = 'sentence-transformers/'+ nm

# 1. Define the SentenceTransformer model.
output_dimension = 256
max_seq_length = 256
train_batch_size = 64
num_epochs = 10


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
df_ip = pd.read_csv(data_file, compression='gzip', sep="\t")
df_ip['simGIC'] = df_ip['simGIC']/100.0 # normalize similarity score values (per the max SimGIC value)
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


#train_loss = losses.CosineSimilarityLoss(model=model)
train_loss = losses.CoSENTLoss(model)

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
    learning_rate = 2e-05,
    warmup_ratio=0.000001,
    #warmup_steps=5000,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=5,
    logging_steps=500,
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

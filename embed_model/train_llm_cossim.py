import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import math
import sys
import os

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

data_dir = "./data/"

model_dir = "./model/"
# check if the directory exists, if not, create it. 
if not os.path.isdir(model_dir): 
    os.makedirs(model_dir)

# load pretrained model
sem_model = SentenceTransformer('nli-distilroberta-base-v2')

# Load the CSV file into a pandas dataframe
df_ip = pd.read_csv(data_dir + 'pairwise-sim_sbert_inputdata.tsv.gz', 
                    compression='gzip', sep='\t')

def sample_prows(data, perc):
    return data.head(int(data.shape[0]*perc))

perc = 0.75
train = sample_prows(df_ip, perc)
test = df_ip.iloc[max(train.index):]

# Extract each pair of descriptions and labels columns as lists
desc_1 = train['desc_1'].tolist()
desc_2 = train['desc_2'].tolist()
labels = train['simGIC'].tolist()

test_desc_1 = test['desc_1'].tolist()
test_desc_2 = test['desc_2'].tolist()
test_labels = test['simGIC'].tolist()

# Create training and test dataset
train_examples = []; trn_length = train.shape[0]
for i in range(trn_length):
  train_examples.append(InputExample(texts=[desc_1[i], desc_2[i]], 
                                     label = labels[i]))
print("training data added to InputExample")

batch_size=128
num_workers=4
num_epochs = 10

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

train_loss = losses.CosineSimilarityLoss(model=sem_model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

model_save_path = (model_dir + 'nli-distilroberta-base-v2' + "-" + 
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

evaluator = EmbeddingSimilarityEvaluator(test_desc_1, test_desc_2, test_labels)

# Finetune model
sem_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=10000,
    warmup_steps=warmup_steps,
    output_path=model_save_path)

# Evaluate model on test data
model = SentenceTransformer(model_save_path)

evaluator(model, output_path = model_save_path)



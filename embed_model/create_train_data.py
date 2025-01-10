import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys

# python embed_model/create_train_data.py extracted-descriptions.tsv pairwise-sim.tsv.gz {percentage} data_{percentage}p_TRAINING.tsv.gz data_{percentage}p_ALL_NON_TRAIN.tsv.gz data_{percentage}p_NON_OVERLAP.tsv.gz

descriptions_path = sys.argv[1]
pairwise_scores_path = sys.argv[2]
perc = int(sys.argv[3])
training_data_out_path = sys.argv[4]
all_non_training_data_out_path = sys.argv[5]
non_overlap_data_out_path = sys.argv[6]

# Load character state descriptions
pheno_desc = pd.read_csv(descriptions_path, sep='\t', header=None, names=['character', 'state', 'description'])
print("Descriptions loaded!")

# Get unique character IDs
ch_ids = pheno_desc['character'].unique()
num_ch_ids = len(ch_ids)
print(f"There are {num_ch_ids} unique character ids")

# Partition input data based on user-definied % "perc" of training data
num_characters = int(len(ch_ids) * perc/100)

# Ensure at least one sample is selected
num_characters = max(1, num_characters)

# Randomly select the training and test num_characters IDs per perc
selected_character_ids = np.random.choice(ch_ids, num_characters, replace=False)

pheno_desc_selected = pheno_desc[pheno_desc['character'].isin(selected_character_ids)]

pheno_desc_test = pheno_desc[~pheno_desc['character'].isin(selected_character_ids)]

# Load pairwise scores
input_pairwise_scores = pd.read_csv(pairwise_scores_path, compression='gzip', header=None, sep='\t')
print("Pairwise ids and scores loaded!")

input_pairwise_scores.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC']
print(input_pairwise_scores)
nonDup_org = len(input_pairwise_scores.duplicated(subset=['id_1','id_2'], keep='first'))
print("Total number of pairs in input data, pairwise-sim.tsv.gz: ", len(input_pairwise_scores))
print("Number of non-duplicate pairs in input data, pairwise-sim.tsv.gz: ", nonDup_org)
print("Number of duplicate pairs in input data, pairwise-sim.tsv.gz: ", len(input_pairwise_scores)-nonDup_org)

input_pairwise_scores['order'] = range(len(input_pairwise_scores))

# Join scores on state to bring in text descriptions
merge1 = pd.merge(input_pairwise_scores, pheno_desc, left_on='id_1', right_on='state').drop(labels='state', axis='columns')
merge1.columns = ['state_1', 'state_2', 'maxIC', 'jaccard', 'simGIC', 'order', 'character_1', 'desc_1']
merge2 = pd.merge(merge1, pheno_desc, left_on='state_2', right_on='state').drop(labels='state', axis='columns')
merge2.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC', 'order', 'character_1', 'desc_1', 'character_2', 'desc_2']

# Partition dataset by selected character ids
train_data = merge2[merge2['character_1'].isin(selected_character_ids) & merge2['character_2'].isin(selected_character_ids)]
non_train_data = merge2[~merge2['character_1'].isin(selected_character_ids) | ~merge2['character_2'].isin(selected_character_ids)]
non_overlap_data = merge2[~merge2['character_1'].isin(selected_character_ids) & ~merge2['character_2'].isin(selected_character_ids)]

# Save combined data
train_data.to_csv(training_data_out_path,
                          compression='gzip', sep='\t')
print("Total pairs in training data: ", train_data.shape[0])
non_train_data.to_csv(all_non_training_data_out_path,
                          compression='gzip', sep='\t')
print("Total pairs in left out data: ", non_train_data.shape[0])
non_overlap_data.to_csv(non_overlap_data_out_path,
                          compression='gzip', sep='\t')
print("Total pairs in non-overlapping data: ", non_overlap_data.shape[0])

print("combined data saved")

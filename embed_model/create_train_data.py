import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Load character state descriptions
pheno_desc = pd.read_csv('extracted-descriptions.tsv', sep='\t')
print("descriptions loaded!")

# Get unique character IDs
ch_ids = pheno_desc['?character'].unique()

# Retain only unique character IDs in pheno_desc
pheno_desc_unq = pheno_desc.drop_duplicates(subset=['?character'])

# Partition input data based on user-definied % "perc" of training data
perc = 10
num_samples = int(len(ch_ids) * perc/100)

# Ensure at least one sample is selected
num_samples = max(1, num_samples)

# Randomly select the training and test IDs per perc
Xp_ids = np.random.choice(ch_ids, num_samples, replace=False)

pheno_desc_Xp = pheno_desc_unq[pheno_desc_unq['?character'].isin(Xp_ids)] 

pheno_desc_Xp_test = pheno_desc_unq[~pheno_desc_unq['?character'].isin(Xp_ids)] 

# Load pairwise scores
df1 = pd.read_csv('pairwise-sim.tsv.gz', compression='gzip', header=None, sep='\t')
print("pairwise ids and scores loaded!")

df1.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC']
nonDup_org = len(df1.duplicated(subset=['id_1','id_2'], keep='first'))
print("Total number of pairs in input data, pairwise-sim.tsv.gz: ", len(df1))
print("Number of non-duplicate pairs in input data, pairwise-sim.tsv.gz: ", nonDup_org)
print("Number of duplicate pairs in input data, pairwise-sim.tsv.gz: ", len(df1)-nonDup_org)

df1['order'] = range(len(df1))

# Define function to create training/test data 
def create_subset(pheno_desc, pairwise_scores):
    
    idStr = pheno_desc_Xp_test['?iri'].values

    pheno_desc_Xp_test.insert(loc = 0, column = 'id_1', value = idStr)

    pheno_desc_Xp_test.columns = ['id_1', 'id_2', 'trait_desc', 'character_id']

    merged_df_2 = pd.merge(df1[['order','id_1', 'id_2', 'simGIC']], 
                           pheno_desc_Xp_test, on='id_2', how='inner')

    merged_df_2.sort_values('order', inplace=True)

    merged_df_2.set_index("order", inplace = True)

    merged_df_2.columns = ['id_1_x', 'id_2', 'simGIC_2', 'id_1_y', 'desc_2','char_id_2']

    id_1_df = df1[['id_1','id_2','simGIC']].merge(pheno_desc_Xp_test, on = 'id_1', 
                                                  how = "inner")

    id_1_df.columns = ['id_1', 'id_2_x', 'simGIC_1', 'id_2_y', 'desc_1','char_id_1']

    id12_desc12_simGIC = pd.concat([id_1_df[['id_1', 'desc_1', 'simGIC_1','char_id_1']], 
                                merged_df_2[['id_2', 'desc_2','char_id_2']]], axis = 1)
    
    id12_desc12_simGIC = id12_desc12_simGIC.dropna()
    
    return id12_desc12_simGIC

# Save combined data
subset = "TRAINING"
id12_desc12_simGIC = create_subset(pheno_desc = pheno_desc_Xp, pairwise_scores = df1)
id12_desc12_simGIC.to_csv(f"id12_desc12_simGIC_{perc}p_{subset}.tsv.gz",
                          compression='gzip', sep='\t')
print("Total pairs in training data: ", id12_desc12_simGIC.shape[0])

subset = "TEST"
id12_desc12_simGIC_TEST = create_subset(pheno_desc = pheno_desc_Xp_test, pairwise_scores = df1)
id12_desc12_simGIC_TEST.to_csv(f"id12_desc12_simGIC_{perc}p_{subset}.tsv.gz",
                               compression='gzip', sep='\t')
print("Total pairs in test data: ", id12_desc12_simGIC_TEST.shape[0])

print("combined data saved")
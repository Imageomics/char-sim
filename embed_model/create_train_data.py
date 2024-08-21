import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

pheno_desc = pd.read_csv('extracted-descriptions.tsv', sep='\t')
print("descriptions loaded!")

idStr = pheno_desc['?id'].values
print("Number of unique trait descriptions: ", 
      len(pheno_desc.duplicated(keep='first')==True))

for i in range(0, len(idStr)):
    idStr[i] = ''.join(('<', idStr[i], '>'))

pheno_desc.insert(loc = 0, column = 'id_1', value = idStr)

pheno_desc.columns = ['id_1', 'id_2', 'trait_desc']

df1 = pd.read_csv('pairwise-sim.tsv.gz', compression='gzip', header=None, sep='\t')
print("pairwise ids and scores loaded!")

df1.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC']
nonDup_org = len(df1.duplicated(subset=['id_1','id_2'], keep='first'))
print("Total number of pairs in input data, pairwise-sim.tsv.gz: ", len(df1))
print("Number of non-duplicate pairs in input data, pairwise-sim.tsv.gz: ", nonDup_org)
print("Number of duplicate pairs in input data, pairwise-sim.tsv.gz: ", len(df1)-nonDup_org)

df1['order'] = range(len(df1))
merged_df_2 = pd.merge(df1[['order','id_1', 'id_2', 'simGIC']], pheno_desc, on='id_2', how='inner')

merged_df_2.sort_values('order', inplace=True)

merged_df_2.set_index("order", inplace = True)

merged_df_2.columns = ['id_1_x', 'id_2', 'simGIC_2', 'id_1_y', 'desc_2']

id_1_df = df1[['id_1','id_2','simGIC']].merge(pheno_desc, on = 'id_1', how = "inner")

id_1_df.columns = ['id_1', 'id_2_x', 'simGIC_1', 'id_2_y', 'desc_1']

id12_desc12_simGIC = pd.concat([id_1_df[['id_1', 'desc_1', 'simGIC_1']], 
                                merged_df_2[['id_2', 'desc_2']]], axis = 1)
descunq=len(id12_desc12_simGIC.duplicated(subset = ['desc_1','desc_2'], keep='first')==True)

print("Total number of pairs in train data, id12_desc12_simGIC: ", len(id12_desc12_simGIC))
print("Number of non-duplicate pairs in train data, id12_desc12_simGIC: ", descunq)
print("Number of duplicate pairs in input data, pairwise-sim.tsv.gz: ", len(id12_desc12_simGIC)-descunq)

id12_desc12_simGIC.to_csv("id12_desc12_simGIC.tsv.gz", compression='gzip', sep='\t')

print("combined data saved")

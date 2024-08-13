import warnings
warnings.filterwarnings('ignore')

import pandas as pd

pheno_desc = pd.read_csv('/extracted-descriptions.tsv', sep='\t')
print("descriptions loaded!")

idStr = pheno_desc['?id'].values

for i in range(0, len(idStr)):
    idStr[i] = ''.join(('<', idStr[i], '>'))

pheno_desc.insert(loc = 0, column = 'id_1', value = idStr)

pheno_desc.columns = ['id_1', 'id_2', 'trait_desc']

df1 = pd.read_csv('/pairwise-sim.tsv.gz', compression='gzip', header=None, sep='\t')
print("pairwise ids and scores loaded!")

df1.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC']

df1['order'] = range(len(df1))
merged_df_2 = pd.merge(df1[['order','id_1', 'id_2', 'simGIC']], pheno_desc, on='id_2', how='inner')

merged_df_2.sort_values('order', inplace=True)

merged_df_2.set_index("order", inplace = True)

merged_df_2.columns = ['id_1_x', 'id_2', 'simGIC_2', 'id_1_y', 'desc_2']

id_1_df = df1[['id_1','id_2','simGIC']].merge(pheno_desc, on = 'id_1', how = "inner")

id_1_df.columns = ['id_1', 'id_2_x', 'simGIC_1', 'id_2_y', 'desc_1']

id12_desc12_simGIC = pd.concat([id_1_df[['id_1', 'desc_1', 'simGIC_1']], 
                                merged_df_2[['id_2', 'desc_2']]], axis = 1)

id12_desc12_simGIC.to_csv("/id12_desc12_simGIC.tsv.gz", compression='gzip', sep='\t')

print("combined data saved")
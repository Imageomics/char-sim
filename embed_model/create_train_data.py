import pandas as pd

pheno_desc = pd.read_csv('extracted-descriptions.gz', compression='gzip', header=None, sep='\t')

df1 = pd.read_csv('pairwise-sim.tsv.gz', compression='gzip', header=None, sep='\t')
df1.columns = ['id_1', 'id_2', 'maxIC', 'jaccard', 'simGIC']
id1_str = inputString = pheno_desc['?id'].values

for i in range(0, len(inputString)):
    id1_str[i] = ''.join(('<', inputString[i], '>'))

pheno_desc.insert(loc = 0, column = 'id_1', value = id1_str)
pheno_desc.columns = ['id_1', 'id_2', 'trait_desc']

id_1_df = df1.merge(pheno_desc, on = 'id_1', how="inner")
id_2_df = df1.merge(pheno_desc, on = 'id_2', how="inner")
id_12_df = pd.concat([id_1_df[['id_1','trait_desc']], 
                      id_2_df[['id_2','trait_desc','simGIC','maxIC','jaccard']]], axis=1)
id_12_df.columns = ['id_1', 'desc_1', 'id_2', 'desc_2', 'simGIC','maxIC','jaccard']

id_12_df.to_csv('pairwise-sim_sbert_inputdata.tsv.gz', compression='gzip', sep='\t')

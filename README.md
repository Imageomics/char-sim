# char-sim
This is the repository for the [Trait2Vec model](https://huggingface.co/imageomics/trait2vec) and the [Character Similarity dataset](https://huggingface.co/datasets/imageomics/char-sim-data). It contains the code used for training and the evaluation of Trait2Vec (testing and visualizing embeddings). Additionally, we include a collection of scripts for forming, evaluating, and visualizing the character similarity data created alongside it. 

[Paper](TBD) | [Model](https://huggingface.co/imageomics/trait2vec) | [Data](https://huggingface.co/datasets/imageomics/char-sim-data)
---

Trait2Vec is a Sentence Transformer model trained on a new character similarity dataset that encodes the similarity between textual trait descriptions of the [Phenoscape knowledgebase](https://kb.phenoscape.org/).

## Table of Contents

1. [Model](#model)
2. [Data](#data)
3. [Paper](#paper)(TBD)
4. [Citation](#citation)(TBD)

## Model

The Trait2Vec model is a [Sentence Transformer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) pre-trained with the [CosENT](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) objective. The dependencies are listed in [`train_environement.yaml`](https://github.com/Imageomics/char-sim/blob/main/train_environment.yaml).

To train a Trait2Vec model on all the dataset, please change directory to this repo and run:
```
conda env create -f train_environment.yaml
conda activate snakemake_env
python train_model.py
```
The above will estimate a Trait2Vec model and save it to `outputs/full_data/model` directory. We also provide a pre-trained model in huggingface [Trait2Vec](https://huggingface.co/imageomics/trait2vec). 

To train a taxon-specific Trait2Vec model change the dataset parameter. The next line will estimate a Trait2Vec model, with the `characiformes` dataset, and save it to `outputs/characiformes/model` directory
```
python train_model.py --dataset characiformes
```

## Data

Trait2Vec was trained on the [Character Similarity dataset](https://huggingface.co/datasets/imageomics/char-sim-data).
The data is a collection of textual trait description pairs and the corresponding Jaccard, maxIC and SimGIC ontology-based similarities. The ontological representations of the corresponding traits, that induces the similarity, is extracted from the [Phenoscape knowledgebase](https://kb.phenoscape.org/). The pipeline to extract the data from Phenoscape and process it is listed in the [Snakefile](https://github.com/Imageomics/char-sim/blob/main/Snakefile). We recommend to load the data directly from the [Hugging Face repo](https://huggingface.co/datasets/imageomics/char-sim-data). After downloading, the datasets should be listed under the data folder. `all` corresponds to the entire set of trait descriptor pairs. `characiformes`, `siluriformes`, `cypriniformes`, `gymnotiformes` correspond to taxon specific trait descriptors. Under each folder, `data_{percentage}p_TRAINING.tsv.gz` and `data_{percentage}p_NON_OVERLAP.tsv.gz` correspond to the train and test splits, respectively. Please see the [Character Similarity dataset](https://huggingface.co/datasets/imageomics/char-sim-data) repo for more details.

<h2 id="paper">Paper, Website, and Docs</h2>

TBD

## Citation

TBD
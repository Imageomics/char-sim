# mamba create -c conda-forge -c bioconda -n snakemake snakemake
# mamba activate snakemake
# snakemake -c2 --show-failed-logs --use-singularity --use-conda pairwise-sim.tsv.gz

data_repo = "https://huggingface.co/datasets/imageomics/phenoscape-character-eqs/resolve/main"

rule retrieve_annotations_file:
    output:
        "phenex-data-merged.ofn.gz"
    container:
        "docker://obolibrary/odkfull:v1.5"
    shell:
        "curl -L -O {data_repo}/{output}"

rule retrieve_tbox_file:
    output:
        "phenoscape-kb-tbox-classified.ttl.gz"
    container:
        "docker://obolibrary/odkfull:v1.5"
    shell:
        "curl -L -O {data_repo}/{output}"

rule convert_ofn_gz_to_ttl:
    input:
        "{ontology}.ofn.gz"
    output:
        "{ontology}.ttl"
    container:
        "docker://obolibrary/robot:v1.9.4"
    shell:
        "robot convert -i {input} -o {output}"

rule extract_descriptions:
    input:
        "phenex-data-merged.ttl",
        "phenoscape-kb-tbox-classified.ttl.gz",
        "sparql/extract-descriptions.rq"
    output:
        "extracted-descriptions.tsv"
    container:
        "docker://stain/jena:5.1.0"
    shell:
        "arq --data phenex-data-merged.ttl --data phenoscape-kb-tbox-classified.ttl.gz --query sparql/extract-descriptions.rq --results tsv | sed -E 's/^\"//' | sed -E 's/\\t\"/\\t/g' | sed -E 's/\"$//' | sed -E 's/\\\\\"/\"/g' | tail -n +2 >{output}"

rule extract_annotations:
    input:
        "phenex-data-merged.ttl",
        "phenoscape-kb-tbox-classified.ttl.gz",
        "sparql/descriptions-to-ontology.rq"
    output:
        "annotations.tsv"
    container:
        "docker://stain/jena:5.1.0"
    shell:
        "arq --data phenex-data-merged.ttl --data phenoscape-kb-tbox-classified.ttl.gz --query sparql/descriptions-to-ontology.rq --results tsv | sed -E 's/^\"//' | sed -E 's/\"\\t\"/\\t/g' | sed -E 's/\"$//' | sed -E 's/\\\\\"/\"/g' | tail -n +2 >{output}"

rule convert_ttl_gz_to_souffle_tsv:
    input:
        "{rdf}.ttl.gz"
    output:
        "{rdf}.facts"
    container:
        "docker://stain/jena:5.1.0"
    shell:
        "{{ riot -q --nocheck --output ntriples {input} || true; }} | sed 's/ /\\t/' | sed 's/ /\\t/' | sed 's/ \\.$//' >{output}"

rule subsumptions_closure:
    input:
        "phenoscape-kb-tbox-classified.facts",
        "scripts/subsumptions.dl"
    output:
        "subsumptions.tsv"
    container:
        "docker://obolibrary/odkfull:v1.5"
    shell:
        "souffle -c scripts/subsumptions.dl"

rule compute_pairwise_similarity:
    input:
        "annotations.tsv",
        "subsumptions.tsv"
    output:
        "pairwise-sim.tsv.gz"
    container:
        "docker://virtuslab/scala-cli:1.3.0"
    shell:
        "scala-cli run --server=false --java-opt -Xmx48G scripts/sim.sc -- {input} {output}"

rule create_train_data:
    input:
        "extracted-descriptions.tsv",
        "pairwise-sim.tsv.gz",
        "embed_model/create_train_data.py"
    output:
        "data_{percentage}p_TRAINING.tsv.gz",
        "data_{percentage}p_ALL_NON_TRAIN.tsv.gz",
        "data_{percentage}p_NON_OVERLAP.tsv.gz"
    conda:
        "environment.yaml"
    shell:
        "python embed_model/create_train_data.py extracted-descriptions.tsv pairwise-sim.tsv.gz {wildcards.percentage} data_{wildcards.percentage}p_TRAINING.tsv.gz data_{wildcards.percentage}p_ALL_NON_TRAIN.tsv.gz data_{wildcards.percentage}p_NON_OVERLAP.tsv.gz"

rule train_model:
    input:
        data="data_{percentage}p_TRAINING.tsv.gz",
        script="embed_model/train_mpnet_v2.py"
    output:
        output_dir=directory("output-{percentage}p")
    conda:
        "train_environment.yaml"
    shell:
        "mkdir {output.output_dir}; python {input.script} {input.data} {output.output_dir}"

rule convert_ofn_to_ttl:
    input:
        "{ontology}.ofn"
    output:
        "{ontology}.ttl"
    container:
        "docker://obolibrary/robot:v1.9.4"
    shell:
        "robot convert -i {input} -o {output}"

rule extract_descriptions:
    input:
        "phenex-data-merged.ttl",
        "sparql/extract-descriptions.rq"
    output:
        "extracted-descriptions.tsv"
    container:
        "docker://stain/jena:4.8.0"
    shell:
        "arq --data phenex-data-merged.ttl --results tsv --query sparql/extract-descriptions.rq | sed -E 's/^\"//' | sed -E 's/\"\\t\"/\\t/' | sed -E 's/\"$//' | sed -E 's/\\\\\"/\"/g' >{output}"

rule extract_annotations:
    input:
        "phenex-data-merged.ttl",
        "sparql/descriptions-to-ontology.rq"
    output:
        "annotations.tsv"
    container:
        "docker://stain/jena:4.8.0"
    shell:
        "arq --data phenex-data-merged.ttl --query sparql/descriptions-to-ontology.rq --results tsv | tail -n +2 >{output}"

rule convert_ttl_to_souffle_tsv:
    input:
        "{rdf}.ttl"
    output:
        "{rdf}.facts"
    container:
        "docker://stain/jena:4.8.0"
    shell:
        "riot -q --nocheck --output ntriples {input} | sed 's/ /\\t/' | sed 's/ /\\t/' | sed 's/ \.$//' >{output}"

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

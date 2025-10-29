#!/bin/bash

mkdir -p data
if [ -f data/msmarco-docs.tsv.gz ]; then
    echo "File data/msmarco-docs.tsv.gz already exists. Skipping download."
    exit 0
fi
wget -O data/msmarco-docs.tsv.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
# Search Engine Architecture

## Setup Instructions

Install poetry if you haven't already. You can find installation instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).
To set up the environment, run:

```bash
cd cs-search-engine-architecture
poetry install
python3 -m spacy download en_core_web_sm
```

To activate the virtual environment, use:

```bash
source $(poetry env info --path)/bin/activate
```

## Run

Download the required dataset by executing:

```bash
bash download.sh
```

To run the main application, use:

```bash
(poetry run) python3 setup.py build_ext && python3 -m sea.main
```


## Project Structure

- `data/`: Directory for storing datasets.
- `build/`: Directory for build artifacts as this project includes Cython modules.
- `sea/`: Main package containing source code.

## TODO

- query "cat cat cat", positionen die wir schon benutzt haben nicht nochmal benutzen
- fix phrase "k"/distance for chars in phrases, dynamix to word lengths of query words/the phrase
- fix tiered serving
- fix bm25 score thresholds

## Learning To Rank

### Feature Selection

1. BM25 Score Body
2. BM25 Score Title
3. TF-IDF Cosine Similarity (Query and Title)
4. Document Length Body
5. Document Length Title
6. First Occurrence Body / Normalizes by Body Length
7. First Occurrence Title / Normalizes by Title Length
8. In Title Indicator (1 if present, 0 otherwise)
9. Bigram Coverage Body
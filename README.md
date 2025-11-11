# Search Engine Architecture

## Setup Instructions
Install poetry if you haven't already. You can find installation instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).
To set up the environment, run:
```bash
cd cs-search-engine-architecture
poetry install
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
(poetry run) python -m sea.main
```
or 
```bash
(poetry run) sea
```

## Project Structure
- `data/`: Directory for storing datasets.
- `build/`: Directory for build artifacts as this project includes Cython modules.
- `sea/`: Main package containing source code.

## TODO
query "the", query "and"
query "cat cat cat", positionen die wir schon benutzt haben nicht nochmal benutzen
# Search Engine Architecture

## Setup Instructions
The project uses Poetry for dependency management. To set up the environment, run:
```bash
poetry install
```
To activate the virtual environment, use:
```bash
source $(poetry env info --path)/bin/activate
```

## Project Structure
- `data/`: Directory for storing datasets.
- `build/`: Directory for build artifacts as this project includes Cython modules.
- `sea/`: Main package containing source code.
import os
import time
from sea.indexer import Indexer
from sea.corpus import Corpus, py_document_processor
from sea.engine import Engine
import shutil

NAME = "100k"
INDEX_PATH = "./data/indices"
EMBEDDINGS_PATH = f"./data/embeddings"
MODEL_PATH = "./data/models"
DATASET = "./data/msmarco-docs.tsv"
MAX_DOCUMENTS = 3_213_835
PARTITION_SIZE = 30_000


def bold_string(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def underline_string(s: str) -> str:
    return f"\033[4m{s}\033[0m"


def color_string(s: str, color_code: int) -> str:
    """
    Color codes:
    Black: 30
    Red: 31
    Green: 32
    Yellow: 33
    Blue: 34
    Magenta: 35
    Cyan: 36
    White: 37
    Bright Black: 90
    Bright Red: 91
    Bright Green: 92
    Bright Yellow: 93
    Bright Blue: 94
    Bright Magenta: 95
    Bright Cyan: 96
    Bright White: 97
    """
    return f"\033[{color_code}m{s}\033[0m"


def index():

    inp = input(f"Do you really want to (re)build the index at {os.path.join(INDEX_PATH, NAME)}? This will delete any existing index. (y/n): ")
    if inp.lower() != "y":
        return
    index_path = os.path.join(INDEX_PATH, NAME)
    if os.path.exists(index_path):
        shutil.rmtree(index_path, ignore_errors=True)

    indexer = Indexer(index_path, DATASET, MAX_DOCUMENTS, PARTITION_SIZE)
    indexer.build()


def serve():

    print("Loading...")
    start = time.time()
    engine = Engine(NAME, INDEX_PATH, EMBEDDINGS_PATH, MODEL_PATH)
    end = time.time()
    print(f"Loaded in {(end - start) * 1000:.4f} milliseconds.")

    mode = "exact"
    ltr_enabled = True
    while True:
        print("type 'mode:exact', 'mode:semantic', 'mode:combined', to select search mode, 'enable:ltr', 'disable:ltr' or 'exit' to quit.")
        query = input(
            f"({color_string(mode, 31)}, {color_string("↓"+('ltr' if ltr_enabled else ('bm25' if mode == 'exact' else '∡')), 31)}) {bold_string(color_string('search', 31))}: "
        )
        if query.lower() == "exit":
            break
        elif query.lower() == "mode:exact":
            mode = "exact"
            print("- Switched to exact search mode.")
            continue
        elif query.lower() == "mode:semantic":
            mode = "semantic"
            print("- Switched to semantic search mode.")
            continue
        elif query.lower() == "mode:combined":
            if not ltr_enabled:
                ltr_enabled = True
                print("- Enabled learning-to-rank since it is required for combined mode.")
            mode = "combined"
            print("- Switched to combined search mode.")
            continue
        elif query.lower() == "enable:ltr":
            ltr_enabled = True
            print("- Enabled learning-to-rank.")
            continue
        elif query.lower() == "disable:ltr":
            if mode == "combined":
                mode = "exact"
                print("- Switched to exact search mode since learning-to-rank is required for combined mode.")
            ltr_enabled = False
            print("- Disabled learning-to-rank.")
            continue
        elif "mode:" in query.lower():
            print("- Unknown mode. Available modes: exact, semantic, combined.")
            continue
        elif "enable:" in query.lower() or "disable:" in query.lower():
            print("- Unknown option. Available options: ltr.")
            continue

        start = time.time()

        if mode == "exact":
            results, sources, semantic_scores = engine.exact_search(query, pre_select_k=50, top_k=10, ltr_enabled=ltr_enabled)
        elif mode == "semantic":
            results, sources, semantic_scores = engine.semantic_search(query, pre_select_k=50, top_k=10, ltr_enabled=ltr_enabled)
        elif mode == "combined":
            results, sources, semantic_scores = engine.combined_search(query, exact_search_preselect_k=50, semantic_search_preselect_k=50, top_k=10)
        end = time.time()
        print(f"- Search took {(end - start) * 1000:.4f} milliseconds.")
        for i, doc in enumerate(results):
            print("-" * os.get_terminal_size().columns)
            print(
                f"{bold_string('ID')}: {doc['id']}"
                + (f" / {bold_string('Source')}: {sources[i]}" if mode == "combined" else "")
                + (f" / {bold_string('BM25 Score')}: {doc['score']:.4f}" if mode != "semantic" else "")
                + (
                    f" / {bold_string('Semantic Score')}: {f'{semantic_scores[i]:.4f}' if semantic_scores[i] > 1e-3 else f'{semantic_scores[i]:2e}'}"
                    if mode != "exact"
                    else ""
                ),
                f"{bold_string('Title')}: {color_string(doc['title'], 32)}",
                f"{bold_string('URL')}: {color_string(doc['url'], 36)}",
                f"{bold_string('Snippet')}: {doc['snippet']}",
                sep="\n",
                end="\n",
            )
        print("=" * os.get_terminal_size().columns)


def main():
    # index()
    serve()


if __name__ == "__main__":
    main()

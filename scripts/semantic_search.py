import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sea.engine import Engine
from sea.corpus import py_document_processor


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_texts(model, tokenizer, device, sentences, matryoshka=None, type="document"):
    empty_doc_ids = [i for i, s in enumerate(sentences) if s.strip() == ""]
    if type == "query":
        sentences = [f"search query: {s}" for s in sentences]
    if type == "document":
        sentences = [f"search_document: : {s}" for s in sentences]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    if matryoshka is not None:
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :matryoshka]
    embeddings[empty_doc_ids] = 0.0
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


MAT_DIM = 64
NAME = "100k"
INDEX_PATH = "./data/indices"
EMBEDDINGS_FILE = f"./data/embeddings/{NAME}.npy"
MODEL_PATH = "./data/models"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.eval()
    model.to(device)

    print("Loading embeddings...")
    embeddings = np.fromfile(EMBEDDINGS_FILE, dtype="float32").reshape(-1, MAT_DIM)
    embeddings = torch.from_numpy(embeddings).to(device)
    num_zero_embeddings = torch.sum(torch.all(embeddings == 0, dim=1)).item()
    print(f"Number of zero embeddings: {num_zero_embeddings}")

    print("Loading search engine...")
    engine = Engine(NAME, INDEX_PATH, EMBEDDINGS_FILE, MODEL_PATH)

    print("=" * 80)
    while True:

        query = input("Enter query: ")
        query_embedding = embed_texts(model, tokenizer, device, [query], matryoshka=MAT_DIM, type="query")
        query_embedding = query_embedding.view(-1)  # Shape: [64]
        cos_sims = torch.mv(embeddings, query_embedding)  # Shape: [N]
        scores, arg_sorted = torch.topk(cos_sims, k=10, largest=True, sorted=True)

        for scr, idx in zip(scores[:3], arg_sorted[:3]):
            print(f"Score: {scr.item()}, ID: {idx.item()}")
            doc = engine.corpus.py_get_document(idx.item(), lowercase=False)
            print(doc["title"])
            print(doc["body"][:200])
            print("-" * 80)
        print("=" * 80)

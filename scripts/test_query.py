import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

from scripts.embed_corpus import embed_texts


MAT_DIM = 64
OUT_FILE = "./data/body_embeddings.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.eval()
    model.to(device)

    embeddings = np.fromfile(OUT_FILE, dtype="float32").reshape(-1, MAT_DIM)
    embeddings = torch.from_numpy(embeddings).to(device)
    print(embeddings.shape)

    query = "yellow flowers"
    encoded_input = tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(device)
    query_embedding = embed_texts(model, tokenizer, device, [query], matryoshka=MAT_DIM, type="query")
    cos_sims = torch.matmul(embeddings, query_embedding.T)
    argmax_idx = torch.argmax(cos_sims).item()
    max_score = cos_sims[argmax_idx].item()
    print(f"Max score: {max_score}", f"Max Id {argmax_idx}")

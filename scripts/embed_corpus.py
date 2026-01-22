import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoTokenizer, AutoModel
import gzip
from tqdm import tqdm

OUT_FILE = "./data/body_embeddings.npy"
BATCH_SIZE = 256
NUM_SAMPLES = 10_000  # 3_213_835
FILE_PATH = "./data/msmarco-docs.tsv.gz"
MAT_DIM = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_texts(model, tokenizer, device, sentences, matryoshka=None, type="document"):
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
        embeddings = embeddings[:, :matryoshka]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def get_embedding_dim(model, tokenizer, device):
    encoded_input = tokenizer(["test"], padding=True, truncation=True, return_tensors="pt")
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings.shape[1]


class TextDataset(Dataset):
    def __init__(self, file_path, num_samples=None):
        self.file_path = file_path
        self.lines = []
        coun = 0
        with gzip.open(self.file_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading Dataset", total=num_samples):
                try:
                    self.lines.append(line.strip().split("\t")[3])
                except Exception:
                    self.lines.append("")
                coun += 1
                if num_samples is not None and coun >= num_samples:
                    break

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True
    )
    model.eval()
    model.to(device)

    embedding_dim = get_embedding_dim(model, tokenizer, device)
    print(f"Embedding dim: {embedding_dim}")

    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    dataset = TextDataset(FILE_PATH, num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    pbar = tqdm(desc="Embedding", total=NUM_SAMPLES // BATCH_SIZE + 1, unit="batch")
    with open(OUT_FILE, "ab") as out_f, torch.inference_mode(), autocast(device_type=device):
        for batch_sentences in dataloader:
            embeddings = embed_texts(model, tokenizer, device, batch_sentences, matryoshka=MAT_DIM)
            out_f.write(embeddings.cpu().numpy().astype("float32").tobytes())
            batch_sentences = []
            pbar.update(1)
        pbar.close()

    embeddings = np.fromfile(OUT_FILE, dtype="float32").reshape(-1, MAT_DIM)
    print(embeddings.shape)

    query = "yellow flowers"
    encoded_input = tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    query_embedding = embed_texts(
        model, tokenizer, device, [query], matryoshka=MAT_DIM, type="query"
    )
    embeddings = torch.from_numpy(embeddings).to(device)
    cos_sims = torch.matmul(embeddings, query_embedding.T)
    argmax_idx = torch.argmax(cos_sims).item()
    max_score = cos_sims[argmax_idx].item()
    print(f"Max score: {max_score}", f"Max Id {argmax_idx}")
    print(f"Document: {dataset[argmax_idx]}")

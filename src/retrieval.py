import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

FAISS_INDEX_FILE = "../../rag_vector_index.faiss"
PDF_CHUNKS_FILE = "../../preprocessed_pdf_chunks.json"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Load FAISS index and chunks
index = faiss.read_index(FAISS_INDEX_FILE)
with open(PDF_CHUNKS_FILE, "r", encoding="utf-8") as f:
    pdf_chunks = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModel.from_pretrained(HF_MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

def embed_query(query: str):
    with torch.no_grad():
        encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        output = model(**encoded)
        embedding = mean_pooling(output, encoded["attention_mask"])
        return embedding.numpy().astype("float32")

def retrieve_documents(query: str, top_k=TOP_K):
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)
    return [pdf_chunks[i]["text"] for i in indices[0]]

import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# ========================
# CONFIG
# ========================
PDF_JSON_FILE = "../../preprocessed_pdf_chunks.json"
OUTPUT_FAISS_INDEX = "../../rag_vector_index.faiss"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load PDF chunks
with open(PDF_JSON_FILE, "r", encoding="utf-8") as f:
    pdf_chunks = json.load(f)

print(f"Total chunks to embed: {len(pdf_chunks)}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModel.from_pretrained(HF_MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

# Generate embeddings
embeddings_list = []

with torch.no_grad():
    for i, chunk in enumerate(pdf_chunks, start=1):
        text = chunk.get("text")
        if not text:
            continue

        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        model_output = model(**encoded_input)
        sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        vector = sentence_embedding.squeeze().numpy().astype("float32")

        embeddings_list.append({
            "doc_id": chunk.get("doc_id"),
            "vector": vector.tolist(),
            "metadata": chunk.get("metadata", {})
        })

        if i % 50 == 0 or i == len(pdf_chunks):
            print(f"Embedded {i}/{len(pdf_chunks)} chunks")

# Create FAISS index
embedding_dim = len(embeddings_list[0]["vector"])
index = faiss.IndexFlatL2(embedding_dim)

vectors = np.array([e["vector"] for e in embeddings_list], dtype="float32")
index.add(vectors)

faiss.write_index(index, OUTPUT_FAISS_INDEX)
print(f"FAISS index saved to {OUTPUT_FAISS_INDEX}")

import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

PDF_JSON_FILE = "preprocessed_pdf_chunks.json"
OUTPUT_FAISS_FOLDER = "faiss_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with open(PDF_JSON_FILE, "r", encoding="utf-8") as f:
    pdf_chunks = json.load(f)
print(f"Total PDF document chunks: {len(pdf_chunks)}")

documents = []
for chunk in pdf_chunks:
    documents.append(
        Document(
            page_content=chunk["text"],
            metadata={
                "policy_name": chunk.get("policy_name", ""),
                "section": chunk.get("section", ""),
                "folder": chunk.get("folder", ""),
                "version": chunk.get("version", ""),
                "effective_date": chunk.get("effective_date", "")
            }
        )
    )
print(f"Documents created: {len(documents)}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
vectorstore.save_local(OUTPUT_FAISS_FOLDER)
print(f"FAISS database saved to: {OUTPUT_FAISS_FOLDER}")

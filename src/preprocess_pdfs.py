import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "D:/New folder (2)/bank_langchain/datas"
PROCESSED_FILE = "processed_pdfs.json"

if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed = json.load(f)
else:
    processed = {}

def preprocess_pdfs(root_folder_path):
    pdf_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for foldername, _, filenames in os.walk(root_folder_path):
        for filename in filenames:
            if not filename.lower().endswith(".pdf"):
                continue

            file_key = filename.lower().strip()
            if processed.get(file_key):
                print(f"Skipping already preprocessed: {filename}")
                continue
            pdf_path = os.path.join(foldername, filename)

            try:
                print(f"Processing: {filename}")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                for idx, chunk in enumerate(chunks):
                    pdf_chunks.append({
                        "policy_name": filename.replace(".pdf", ""),
                        "section": f"Chunk_{idx + 1}",
                        "text": chunk.page_content,
                        "version": "v1.0",
                        "page": chunk.metadata.get("page", "")
                    })
                processed[file_key] = True
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return pdf_chunks

chunks = preprocess_pdfs(PDF_FOLDER)
print(f"\nTotal chunks created: {len(chunks)}")
if len(chunks) > 0:
    with open("preprocessed_pdf_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(chunks)} new chunks")
else:
    print("No new PDFs found. Existing chunk file kept unchanged.")

with open(PROCESSED_FILE, "w") as f:
    json.dump(processed, f, indent=4)

print("\nSaved successfully.")

import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2

nltk.download('punkt')

def preprocess_pdfs(root_folder_path, chunk_size=5):
    """
    Reads all PDFs from root_folder_path, splits text into chunks of sentences,
    and returns a list of chunk dictionaries.
    """
    pdf_chunks = []

    for foldername, _, filenames in os.walk(root_folder_path):
        print(f"\nEntering folder: {foldername}")
        print(f"Files found: {filenames}")
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(foldername, filename)
                with open(pdf_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                sentences = sent_tokenize(text)

                for i in range(0, len(sentences), chunk_size):
                    chunk_text = " ".join(sentences[i:i + chunk_size])
                    chunk_data = {
                        "doc_id": f"{filename.replace('.pdf','')}_CHUNK_{i // chunk_size + 1}",
                        "policy_name": filename.replace(".pdf", ""),
                        "section": f"Chunk_{i // chunk_size + 1}",
                        "text": chunk_text,
                        "version": "v1.0",
                        "effective_date": "2026-01-01",
                        "folder": foldername,
                        "metadata": {}
                    }
                    pdf_chunks.append(chunk_data)

    return pdf_chunks

if __name__ == "__main__":
    root_folder = "../../data/pdfs"  # Adjust path relative to this script
    output_file = "../../preprocessed_pdf_chunks.json"
    chunks = preprocess_pdfs(root_folder)
    print(f"Total chunks created: {len(chunks)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"Saved chunks to {output_file}")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_FOLDER = "faiss_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
vectorstore = FAISS.load_local(
    FAISS_FOLDER,
    embeddings,
    allow_dangerous_deserialization=True
)

def retrieve_documents(query, top_k=5):
    results = vectorstore.similarity_search(
        query=query,
        k=top_k
    )
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        results = retrieve_documents(query)
        print(f"\nRetrieved {len(results)} relevant chunks.")

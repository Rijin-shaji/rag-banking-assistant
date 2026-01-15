# rag-banking-assistant
A RAG-based AI assistant that answers banking and financial queries using document knowledge.

# rag-banking-assistant 

An LLM-powered Retrieval-Augmented Generation (RAG) system for answering banking and finance questions from PDFs.

---

##  Project Overview

`rag-banking-assistant` is an AI-based question-answering system that allows users to ask questions about **banking and finance topics** such as:
- NEFT, RTGS, IMPS
- Account opening rules
- Minimum balance requirements
- Deposits, loans, leverage, interest rates
- Other banking-related concepts

The system retrieves relevant information from **PDF documents** and generates accurate answers using a **Large Language Model (LLM)**.

---

##  How It Works (RAG Flow)

1. PDFs are loaded and split into text chunks  
2. Text chunks are converted into **vector embeddings**
3. Embeddings are stored in a **vector database**
4. User query is embedded and matched with relevant chunks
5. Retrieved context is passed to the LLM
6. LLM generates a grounded, document-based answer

---

##  Tech Stack

- Python 
- Large Language Model (LLM)
- Embeddings
- Vector Database (FAISS / similar)
- PDF Loader
- LangChain / custom RAG pipeline

---




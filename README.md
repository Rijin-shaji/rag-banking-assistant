Banking RAG Assistant

An Enterprise-Grade Banking Retrieval-Augmented Generation (RAG) Assistant built using Python, LangChain, FAISS, Hugging Face Embeddings, Groq Llama 3.3, GitHub-based document ingestion, and Streamlit.

The system retrieves relevant information from banking documents and generates accurate, context-aware responses using Large Language Models (LLMs).

 Features
Banking-specific question answering
Retrieval-Augmented Generation (RAG)
Automatic document ingestion from GitHub
Incremental PDF processing using SHA tracking
Semantic document retrieval using FAISS
Query correction and contextualization
Conversation memory
Conversation summarization
Streamlit web interface
Groq Llama 3.3 integration
Modular architecture for easy scaling

Programming Language
Python
Frameworks & Libraries
Streamlit
LangChain
FAISS
Hugging Face Transformers
Sentence Transformers
Groq API
PyPDFLoader
Pydantic
Embedding Model
sentence-transformers/all-MiniLM-L6-v2
LLM
Llama 3.3 70B Versatile (Groq)

Document Ingestion Pipeline

The system automatically checks the GitHub repository for banking documents.

Workflow
Fetch PDF files from GitHub
Compare SHA values with previously processed files
Download only new or updated PDFs
Skip already processed files
Store PDFs in the local document folder

This prevents duplicate processing and improves efficiency.

 Document Preprocessing

Banking documents are processed using:

PyPDFLoader
RecursiveCharacterTextSplitter
Chunk Settings
chunk_size = 1000
chunk_overlap = 200

Each chunk contains:

Policy Name
Section Number
Chunk Content
Page Number
Version Information
Retrieval System

The retrieval layer uses:

Hugging Face Embeddings
FAISS Vector Search
Semantic Similarity Search

Workflow:

User Query
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Similarity Search
     │
     ▼
Top Relevant Chunks
 LLM Processing

The retrieved chunks are provided to:

Llama 3.3 70B Versatile (Groq)

Capabilities:

Context-based answering
Query correction
Follow-up question understanding
Conversation summarization

The model is instructed to answer only from the retrieved context.

Streamlit Interface

The application provides a user-friendly web interface.

Features
Ask banking-related questions
Real-time responses
Context-aware conversations
Clean web-based UI

Run:

streamlit run app.py

# RAG Knowledge Search System

## Business Problem
Help sales, product, and servicing teams retrieve accurate answers from enterprise documents.

## Methods
- Document chunking
- TF-IDF vector search baseline
- Cosine similarity retrieval
- Context-grounded answer generation template
- Designed for later upgrade to FAISS, Pinecone, LangChain, or OpenAI embeddings

## Run
```bash
pip install -r requirements.txt
python rag_search.py
```

## Resume Bullet
Built a retrieval-augmented search pipeline that chunks internal knowledge documents, indexes them with vector representations, retrieves relevant context using cosine similarity, and generates grounded responses for enterprise product and servicing questions.

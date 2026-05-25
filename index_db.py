from dotenv import load_dotenv
from langsmith.utils import get_api_key
load_dotenv()
import os
from typing import Any,Dict,List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import chromadb
from langchain_chroma import Chroma

CHROMA_DIR       = "chroma_store"
COLLECTION_NAME  = "New_collection"
BATCH_SIZE       = 20

def load_documents(folder: str) -> List[Document]:
    documents = []
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{folder}'. Run download_data.py first.")

    print(f"Loading {len(txt_files)} files from '{folder}'...")
    for fname in txt_files:
        path = os.path.join(folder, fname)
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = fname
                doc.metadata["source"]   = fname.replace(".txt", "")
            documents.extend(docs)
            print(f"  Loaded: {fname} ({len(docs[0].page_content)} chars)")
        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")

    print(f"Total documents loaded: {len(documents)}\n")
    return documents

def semantic_chunk_documents(documents: List[Document], embeddings) -> List[Document]:
    """
    percentile=85 means split when similarity drops below 85th percentile.
    """
    semantic_splitter = SemanticChunker(
        embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=85,   )

    chunks = []
    for doc in documents:
        try:
            doc_chunks = semantic_splitter.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata], )
            chunks.extend(doc_chunks)
        except Exception as e:
            print(f"  Chunking failed for {doc.metadata.get('source')} — {e}, skipping.")

    print(f"Chunking: {len(documents)} pages → {len(chunks)} chunks")
    return chunks


def add_batch(vectorstore: Chroma,batch, batch_num, total_batches):
    try:
        vectorstore.add_documents(batch)  
        print(f"VectorStore Indexing: Successfully added batch {batch_num}/{total_batches} ({len(batch)} documents)")
        return True
    except Exception as e:
        print(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
        return False


def init_chromadb(embeddings) -> Chroma:
    """
    Creates  ChromaDB collection.
    Deletes existing collection if it exists.
    """
    print(f"Connecting to ChromaDB at: '{CHROMA_DIR}'")
    raw_client = chromadb.PersistentClient(path=CHROMA_DIR) #creates new and remove exisiting 
    existing = [c.name for c in raw_client.list_collections()]
    if COLLECTION_NAME in existing:
        raw_client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}' for fresh ingest.")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    print(f"  Collection '{COLLECTION_NAME}' ready.\n")
    return vectorstore


def main():
    all_documents = load_documents('data')
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("Semantic chunking...")
    docs_chunked = semantic_chunk_documents(all_documents, embeddings)

    vectorstore = init_chromadb(embeddings)

    print(f"Indexing {len(docs_chunked)} chunks in batches of {BATCH_SIZE}...")
    batches = [ docs_chunked[i : i + BATCH_SIZE]
        for i in range(0, len(docs_chunked), BATCH_SIZE) ]

    results = [ add_batch(vectorstore, batch, i + 1, len(batches))
        for i, batch in enumerate(batches) ]
    print(f"\nIndexing : {sum(results)}/{len(batches)} batches added succeeded.")

    count = vectorstore._collection.count()
    print(f"Total vectors in ChromaDB: {count}")

    return vectorstore



if __name__ == "__main__":
    main()


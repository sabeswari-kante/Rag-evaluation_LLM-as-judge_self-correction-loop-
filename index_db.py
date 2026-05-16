from dotenv import load_dotenv
from langsmith.utils import get_api_key
load_dotenv()
import os
from typing import Any,Dict,List
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import ( PointStruct, SparseVector, Distance, VectorParams, SearchParams,
                                    SparseVectorParams, HnswConfigDiff, SparseIndexParams  )
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
import certifi
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
import ssl 
from webcrawling import tool_crawl 

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# for qdrant 
COLLECTION_NAME_QDRANT = os.getenv("COLLECTION_NAME_QDRANT")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def qdrant_client_init():
    return QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY)

def setup_collection(client: QdrantClient):
    client.create_collection(
        collection_name=COLLECTION_NAME_QDRANT,
        vectors_config=VectorParams(
                size=1024, distance=Distance.COSINE, 
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200,)
                    ,)
    )
    print("Collection created.")


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


def add_batch(vectorstore,batch, batch_num, total_batches):
    try:
        vectorstore.add_documents(batch)  # sync version (no await)
        print(f"VectorStore Indexing: Successfully added batch {batch_num}/{total_batches} ({len(batch)} documents)")
        return True
    except Exception as e:
        print(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
        return False


def main():
    print("Hello from demo-sample!")
    all_documents = tool_crawl()

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    docs_splitted = semantic_chunk_documents(all_documents, embeddings)

    # qdrant client
    client = qdrant_client_init()
    setup_collection(client)
    print('VectorDB storing.....')

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME_QDRANT,
        embedding=embeddings,            
        vector_name="",    )
        
    BATCH_SIZE = 20
    # add indexing to db
    batches = [docs_splitted[i : i + BATCH_SIZE]
        for i in range(0, len(docs_splitted), BATCH_SIZE) ]

    complete_data_batches = [
        add_batch(vectorstore, batch, i + 1, len(batches))
        for i, batch in enumerate(batches) ]

    print(f'Completed: {sum(complete_data_batches)}/{len(batches)}')

    count =client.count(COLLECTION_NAME_QDRANT)
    print(f'Count: {count}')
    # print(batches)
    return vectorstore


if __name__ == "__main__":
    main()


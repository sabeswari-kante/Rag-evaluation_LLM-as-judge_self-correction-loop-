from dotenv import load_dotenv
from langsmith.utils import get_api_key
load_dotenv()
import os
from typing import Any,Dict,List
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import certifi
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
import ssl 
from webcrawling import tool_crawl 

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()



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
    # embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=200)

    docs_splitted = text_splitter.split_documents(all_documents)
    batch_size = 50 
    batches = [ docs_splitted[i:i+ batch_size] for i in range(0, len(docs_splitted),batch_size)]
    print('VectorDB storing.....')

    # vectordb
    vectorstore = Chroma(persist_directory = 'chroma_db' ,embedding_function= embeddings)
    # vectorstore = PineconeVectorStore(index_name = 'demo', embedding=embeddings)

    # add indexing to db
    complete_data_batches = []
    for i, batch in enumerate(batches):
        temp_= add_batch(vectorstore,batch,i+1, len(batches))
        complete_data_batches.append(temp_)
    print(f'Completed: {sum(complete_data_batches)}/{len(batches)}')
    # print(batches)
    return vectorstore


if __name__ == "__main__":
    main()


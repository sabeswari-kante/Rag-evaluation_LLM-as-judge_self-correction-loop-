from dotenv import load_dotenv
load_dotenv()
import os
from index_db import qdrant_client_init
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = os.getenv("COLLECTION_NAME_QDRANT")
EMBEDDING_DIM = 1024


client = qdrant_client_init()

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print("✓ Old collection deleted")

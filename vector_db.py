from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import ollama
import pandas as pd
import os
import glob
import logging
import chunking

LOCAL_DIRECTORY = r"C:/Users/ankam/Documents/pdf_folder"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_DIR = "./chroma_db"
    
def get_vector_db(chunks):
    # Check if vector DB already exists locally
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        # Load the existing persisted database
        logging.info("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
        )
    else:
        # Pull embedding model and create a new database
        logging.info("Creating new vector database...")
        ollama.pull(EMBEDDING_MODEL)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=VECTOR_STORE_DIR
        )
        vectordb.persist()
        logging.info("Vector database created and persisted locally.")

    return vectordb
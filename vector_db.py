from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import pandas as pd
import os
import glob
import logging

LOCAL_DIRECTORY = r"C:/Users/ankam/Documents/pdf_folder"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_DIR = "./chroma_db"

def ingest_docs(local_directory):
    all_docs =[]
    for pdf_file in glob.glob(os.path.join(local_directory, '*.pdf')):
        loader = PyMuPDFLoader(pdf_file)
        all_docs.extend(loader.load())

    logging.info("Successfully read all docs")
    return all_docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks
    
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
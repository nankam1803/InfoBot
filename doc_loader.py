import logging
import glob
from langchain_community.document_loaders import PyMuPDFLoader

LOCAL_DIRECTORY = r"C:/Users/ankam/Documents/pdf_folder"

def ingest_docs(local_directory):
    all_docs =[]
    for pdf_file in glob.glob(os.path.join(local_directory, '*.pdf')):
        loader = PyMuPDFLoader(pdf_file)
        all_docs.extend(loader.load())

    logging.info("Successfully read all docs")
    return all_docs
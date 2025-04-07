import logging

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size = chunk_size,
        chunk_overlap = int(chunk_size/10))
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks
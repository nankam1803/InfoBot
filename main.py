import os
import glob
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import ollama
import streamlit as st
import vector_db as vdb
import doc_loader

logging.basicConfig(level=logging.INFO)

LOCAL_DIRECTORY = r"C:/Users/ankam/Documents/pdf_folder"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

def setup_qa_chain(vector_db):
    llm = ChatOllama(model=MODEL_NAME)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    prompt_template = """
You are a helpful and friendly university IT helpdesk assistant.

Given the context below, answer the user's question accurately. 
If the user's question is a general greeting, casual conversation, or not related to the provided context, respond in a friendly and conversational manner.

Context:
{context}

User's Question:
{question}

Assistant Response:
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    logging.info("QA chain created successfully.")
    return qa_chain

# Streamlit App

def main():
    st.image("UTLogo.png", width=150)
    st.title("InfoBot")

    # Initialize once
    if "qa_chain" not in st.session_state:
        with st.spinner("Loading and preparing documents..."):
            docs = doc_loader.ingest_docs(LOCAL_DIRECTORY)
            if not docs:
                st.error("No PDFs found in the specified directory.")
                return
            chunks = vdb.split_documents(docs)
            vector_db = vdb.get_vector_db(chunks)
            st.session_state.qa_chain = setup_qa_chain(vector_db)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

    user_input = st.chat_input("How can I help you?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                    answer = response["result"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Optional: Display source documents
                    #with st.expander("Source Documents"):
                       #for doc in response["source_documents"]:
                            #st.write(doc.page_content)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
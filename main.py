from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from langchain_huggingface import HuggingFaceEmbeddings
# Load and process documents
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

def convert_docs_chroma():
    load_dotenv()

    loaders = PyPDFDirectoryLoader("hr_docs")
    loaded_docs = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(loaded_docs)

    # Delete existing Chroma database if it exists
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Create new Chroma vector store
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

    # Persist the database
    vectorstore.persist()

    # Zip the Chroma database directory
    shutil.make_archive("chroma_db", 'zip', "chroma_db")


if __name__=="__main__":
    convert_docs_chroma()
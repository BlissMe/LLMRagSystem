from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import os
import key_param


client = MongoClient(key_param.MONGO_URI)
db = client["Depression_Knowledge_Base"]
collection = db["cbt"]

pdf_folder = "./CBT"
embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 

all_docs = []

#  Loop through all PDFs
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        print(f"Processing: {filename}")
        
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to each chunk
        for i, doc in enumerate(chunks):
            doc.metadata["source"] = filename
            doc.metadata["chunk_index"] = i
        
        all_docs.extend(chunks)

# Store all chunks at once
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents=all_docs,
    embedding=embedding,
    collection=collection,
)

print(f"Successfully embedded and stored {len(all_docs)} chunks from PDFs in MongoDB.")
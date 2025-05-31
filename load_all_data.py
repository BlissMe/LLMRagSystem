from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
import os
import key_param

# MongoDB connection
client = MongoClient(key_param.MONGO_URI)
db = client["Depression_Knowledge_Base"]
collection = db["all_knowledge"]  

# Initialize embedding and splitter
embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Folder structure: each category (e.g. cbt, depression) 
base_folder = "./"
categories = ["CBT", "Depression", "Mindfullness", "PHQ9"]

all_docs = []

for category in categories:
    folder_path = os.path.join(base_folder, category)
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        continue

    print(f"\n Processing category: {category}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            print(f"🔄 Loading: {filename}")

            loader = PyPDFLoader(filepath)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)

            # Add metadata to each chunk
            for i, doc in enumerate(chunks):
                doc.metadata["source"] = filename
                doc.metadata["chunk_index"] = i
                doc.metadata["category"] = category  # important for merged collection

            all_docs.extend(chunks)

# Store all in a single vector index collection
if all_docs:
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=all_docs,
        embedding=embedding,
        collection=collection,
    )
    print(f"\n Embedded and stored {len(all_docs)} chunks into 'all_knowledge' collection.")
else:
    print("\n No documents found to embed.")

client.close()

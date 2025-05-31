from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import key_param


# --- Connect to MongoDB ---
client = MongoClient(key_param.MONGO_URI)
DB_NAME = "Depression_Knowledge_Base"
COLLECTION_NAME = "depression"

collection = client[DB_NAME][COLLECTION_NAME]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default1" 
# --- Create embedding model ---
embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME 
)
print("Connected to existing MongoDB Atlas Vector Search.")


# --- Query to search ---
query = """what is depression? """
print(f"\nSearching for: '{query}'")

# Retrieve top 3 similar documents
results = vectorstore.similarity_search(query, k=3,include_scores=True)
retriever = vectorstore.as_retriever() 
print(f"Found {results} results.")


# Print the results
if results:
    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
else:
    print("No results found.")

client.close()
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_mongodb import MongoDBAtlasVectorSearch
import key_param
from openai import OpenAI


client = MongoClient(key_param.MONGO_URI)
DB_NAME = "Depression_Knowledge_Base"
COLLECTION_NAME = "depression"
collection = client[DB_NAME][COLLECTION_NAME]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default1" 

embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)


vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME 
)
print("Connected to existing MongoDB Atlas Vector Search.")


# --- Query to search ---
query = """what is PHQ-9? """
print(f"\nSearching for: '{query}'")

# Retrieve top 3 similar documents
results = vectorstore.similarity_search(query, k=3,include_scores=True)
retriever = vectorstore.as_retriever() 
print(f"Found {results} results.")

all_results = []
for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(f"Content: {doc.page_content[:500]}...") 
  
      
    all_results.append(doc.page_content[:500])


client.close()


Client1 = OpenAI(api_key=key_param.openai_api_key)

response = Client1.responses.create(
    model="gpt-4.1",
    input=f"This is User query: {query} context: {all_results} ",
    instructions= "your role is to answer user query by referring given context" # Add the context here

)
print("****response: ", response.output_text)

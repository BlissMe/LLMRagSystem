from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from openai import OpenAI
import key_param
from fastapi.middleware.cors import CORSMiddleware

# Creating an instance of the FastAPI application
app = FastAPI()

# CORS so frontend can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_query: str

# Defining a GET endpoint at the root ("/") used for health check
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Defining a POST endpoint to ask a question
@app.post("/ask")
async def ask_question(data: QueryRequest):
    query = data.user_query

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
    
    # Retrieve top 3 similar documents
    results = vectorstore.similarity_search(query, k=3,include_scores=True)
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
    return {"response": response.output_text}

    
    

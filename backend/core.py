import os
from typing import Any
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import logging
import time
from pinecone import Pinecone , ServerlessSpec

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")

)

INDEX_NAME = "chatbot"

def run_llm(query: str) -> Any:
    start_time = time.time()
    
    # Use Ollama's embeddings
    logger.info("Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    logger.info("Setting up Pinecone vector store...")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Use Ollama's chat model
    chat = Ollama(model='salmatrafi/acegpt:13b', temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    result = qa({"query": query})

    end_time = time.time()
    logger.info(f"Query processed in {end_time - start_time} seconds")

    return result

if __name__ == "__main__":
    query = "أنا عايز انقل العداد لحد تاني ايه الاجراءات ؟"
    logger.info(f"Running query: {query}")
    res = run_llm(query=query)
    print(res["result"])
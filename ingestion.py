from dotenv import load_dotenv
import os

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"  # Example model, you can use others
)

class Document:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def custom_text_loader(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Split the file content into separate Q&A pairs
        qna_pairs = content.strip().split('\n\n')
        for pair in qna_pairs:
            if pair.strip():
                q, a = pair.split('\n')
                question = q.replace('Q: "', '').replace('"', '').strip()
                answer = a.replace('A: "', '').replace('"', '').strip()
                documents.append(Document(content=f'{question} {answer}', metadata={'source': file_path}))
    return documents

def ingest_docs():
    loader_path = "formatted_output.txt"
    print("Path exists:", os.path.exists(loader_path))
    
    if not os.path.exists(loader_path):
        print(f"File {loader_path} does not exist.")
        return

    raw_documents = custom_text_loader(loader_path)
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("formatted_output.txt", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="chatbot"
    )
    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATA_FOLDER = "data"

def load_and_split_pdf(pdf_path: Path) -> List:
    """Load PDF and split it into chunks using LangChain."""
    # Load the PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # Add metadata from file path
    show_name = pdf_path.parent.name
    file_name = pdf_path.name
    
    for doc in documents:
        doc.metadata.update({
            "show_name": show_name,
            "episode_title": file_name.rsplit('.', 1)[0],
            "file_name": file_name,
            "file_path": str(pdf_path)
        })
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=75
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def main():
    # Initialize embedding model
    huggingface_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE), #size of embedding dimensions
        )
    else:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE), #size of embedding dimensions
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=huggingface_embeddings,
    )
    
    # Find all PDF files in the data folder and subfolders
    data_path = Path(DATA_FOLDER)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder '{DATA_FOLDER}' not found")
    
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path}")
            chunks = load_and_split_pdf(pdf_path)
            print(f"Split into {len(chunks)} chunks")
            
            # Add to Qdrant
            vector_store.add_documents(documents=chunks)
            
            print(f"Uploaded {len(chunks)} chunks from {pdf_path}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
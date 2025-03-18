from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


QDRANT_URL = os.environ.get("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATA_FOLDER = "data"

EMBEDDING_MODEL= "BAAI/bge-m3"
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY") 
LLM_MODEL= "gpt-4o-mini"
TEMPERATURE= 0.2
MAX_TOKENS= 1000

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)
results = {}
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# cached_embedder = get_cached_embeddings(embedding_model, namespace= COLLECTION_NAME)
            
qdrant_vs = QdrantVectorStore(
    client=client,
    collection_name= COLLECTION_NAME,
    embedding=embedding_model
)
naive_retriever = qdrant_vs.as_retriever(search_kwargs={"k" : 10})
multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=llm)
print('pass1')

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=naive_retriever)
print('pass2')

retriever_list = [naive_retriever, multi_query_retriever, compression_retriever]
equal_weighting = [1/len(retriever_list)] * len(retriever_list)
ensemble_retriever = EnsembleRetriever(retrievers=retriever_list, weights=equal_weighting)
print('pass3')
results[COLLECTION_NAME] = ensemble_retriever.invoke("Explain to me what the recent front burner's US-European alliance episode was about?")
print('pass4')
formatted_results = []
for collection, results in results.items():
    for doc in results:
        formatted_results.append(
            {
            "source": collection,
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        )
print(formatted_results)
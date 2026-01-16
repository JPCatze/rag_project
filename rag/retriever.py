import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.HttpClient(host='localhost', port=8000)

embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",device='cuda')

collection = client.get_or_create_collection(name="test_collection")

def query(input):
    data = input
    if dict:
        embedded_query = embed_model.encode([data], normalize_embeddings=True, prompt_name="query")
        result = collection.query(query_embeddings=embedded_query,n_results=10, include=["documents", "metadatas","distances"])
        values = [i['row'] for i in result['metadatas'][0]]
        zip_dict = dict(zip(result['documents'][0],values))
        data = zip_dict
    return data

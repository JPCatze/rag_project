import pyarrow.parquet as pq
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

client = chromadb.HttpClient(host='localhost', port=8000)

embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",device='cuda')

collection = client.get_or_create_collection(name="test_collection")

table = pq.read_table('data/wiki_text.parquet',columns=['id','title'])
ids = table['id'].to_pylist()
titles = table['title'].to_pylist()

for i in tqdm(range(0, len(titles), 100)):
    batch_titles = titles[i:i+100]
    batch_ids = [str(x) for x in ids[i:i+100]]

    embeddings = embed_model.encode(
        batch_titles,
        normalize_embeddings=True,
        batch_size=50
    )

    collection.upsert(
        ids=batch_ids,
        documents=batch_titles,
        embeddings=embeddings.tolist(),
        metadatas=[{'row': x} for x in batch_ids]
    )
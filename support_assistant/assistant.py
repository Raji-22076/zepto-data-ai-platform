import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "support_assistant/chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name="zepto_policies"
)

def retrieve_documents(question, top_k=3):
    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]

question = input("Ask your question: ")

documents = retrieve_documents(question)

print("\nRelevant information:\n")

for document in documents:
    print(document)
    print("-" * 60)
    
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    client.delete_collection("zepto_policies")
except Exception:
    pass

collection = client.create_collection("zepto_policies")

documents = []
ids = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()
    if text:
        documents.append(text)
        ids.append(file_path.stem)

embeddings = embedding_model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents loaded:", len(documents))
print("Document IDs:", ids)
print("First document:", documents[0])
print("ChromaDB collection created successfully")
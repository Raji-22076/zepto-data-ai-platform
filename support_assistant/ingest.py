from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("support_assistant/docs")
DB_PATH = "support_assistant/chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="zepto_policies"
)

documents = []
ids = []

for file in sorted(DOCS_PATH.glob("*.txt")):
    text = file.read_text(encoding="utf-8")
    documents.append(text)
    ids.append(file.stem)

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents loaded:", len(documents))
print("ChromaDB collection created successfully")
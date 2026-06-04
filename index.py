import chromadb
from ingest import load_chunks

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers"


def build_index():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding and indexing...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
        print(f"  indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    build_index()
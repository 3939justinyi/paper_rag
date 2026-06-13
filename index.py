import chromadb
from ingest import load_chunks

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers"

# Stronger default backend. These MUST match the BGE settings in ask.py: the app
# queries this collection with raw BGE vectors, so the passages here have to be
# embedded with the same model (and, per BGE, WITHOUT the query-only instruction
# prefix -- that prefix is added at query time in ask.py).
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_COLLECTION = "papers_bge_small"


def build_index():
    """Original index: Chroma's built-in all-MiniLM-L6-v2 embeddings (collection
    'papers'). Used by the RAG_EMBEDDING=minilm fallback and the eval baseline."""
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding into '{COLLECTION_NAME}' (MiniLM)...")

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


def build_bge_index():
    """Stronger index used by the app's default backend: BAAI/bge-small-en-v1.5
    (collection 'papers_bge_small'). Passages are embedded without a prefix; the
    query-time instruction prefix lives in ask.py. Skips if already built."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = {getattr(c, "name", c) for c in client.list_collections()}
    if BGE_COLLECTION in existing:
        col = client.get_collection(BGE_COLLECTION)
        if col.count() > 0:
            print(f"'{BGE_COLLECTION}' already built ({col.count()} chunks); skipping.")
            return

    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding into '{BGE_COLLECTION}' ({BGE_MODEL})...")
    model = SentenceTransformer(BGE_MODEL)
    emb = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True, batch_size=64, show_progress_bar=True,
    )

    collection = client.get_or_create_collection(name=BGE_COLLECTION, metadata={"hnsw:space": "cosine"})
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[e.tolist() for e in emb[start:start + batch_size]],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
        print(f"  indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{BGE_COLLECTION}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    build_index()       # MiniLM fallback / eval baseline
    build_bge_index()   # stronger default backend

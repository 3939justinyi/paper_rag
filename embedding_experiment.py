"""Experiment: does a stronger embedding model fix the retrieval bottleneck?

The retrieval eval (retrieval_eval.py) showed the bottleneck is the embedding
model -- the right paper often is not retrieved at all, which a reranker cannot
fix. This re-embeds the SAME chunks with a stronger model and compares retrieval
head-to-head. Same chunks (ids + text from ingest.load_chunks), same questions,
same gold, same cross-encoder reranker -- only the embedding model changes.

Shipped model : all-MiniLM-L6-v2 (Chroma default, 384-d)   -> collection "papers"
Stronger model: BAAI/bge-small-en-v1.5 (384-d, big MTEB jump for retrieval)

BGE is asymmetric: passages are embedded as-is, queries get an instruction prefix.
Chroma's automatic embedding cannot express that, so we embed passages and queries
ourselves and store/query by raw vectors (cosine space).

Run:
    python embedding_experiment.py
"""
import chromadb
import torch
from sentence_transformers import SentenceTransformer

from ingest import load_chunks
from retrieval_eval import (
    EVAL_SET, _norm, K_VALUES, N_CANDIDATES, eval_retriever, _recall_at_k, _mrr,
    get_reranker, base_ranking as minilm_base, reranked_ranking as minilm_reranked,
)

CHROMA_DIR = "chroma_db"
NEW_MODEL = "BAAI/bge-small-en-v1.5"
NEW_COLLECTION = "papers_bge_small"
# BGE v1.5 retrieval instruction, prepended to QUERIES only (not passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_device = "mps" if torch.backends.mps.is_available() else "cpu"
_model = None
_new_col = None


def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(NEW_MODEL, device=_device)
    return _model


def build_index():
    """Build (or reuse) the BGE collection from the exact same chunks."""
    client = chromadb.PersistentClient(CHROMA_DIR)
    names = [getattr(c, "name", c) for c in client.list_collections()]
    if NEW_COLLECTION in names:
        col = client.get_collection(NEW_COLLECTION)
        if col.count() > 0:
            print(f"Reusing existing '{NEW_COLLECTION}' ({col.count()} chunks).")
            return col
    else:
        col = client.create_collection(NEW_COLLECTION, metadata={"hnsw:space": "cosine"})

    print("Loading + chunking PDFs (same chunking as the app)...")
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} passages with {NEW_MODEL} on {_device}...")
    emb = model().encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True, batch_size=64, show_progress_bar=True,
    )
    batch = 500
    for i in range(0, len(chunks), batch):
        s = slice(i, i + batch)
        col.add(
            ids=[c["id"] for c in chunks[s]],
            embeddings=[e.tolist() for e in emb[s]],
            documents=[c["text"] for c in chunks[s]],
            metadatas=[{"source": c["source"]} for c in chunks[s]],
        )
    print(f"Built '{NEW_COLLECTION}' with {col.count()} chunks.")
    return col


def bge_base_ranking(query, n=N_CANDIDATES):
    """Plain BGE vector retrieval (query gets the instruction prefix)."""
    qemb = model().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0].tolist()
    res = _new_col.query(query_embeddings=[qemb], n_results=n, include=["documents", "metadatas"])
    return list(zip(res["documents"][0], [m["source"] for m in res["metadatas"][0]]))


def bge_reranked_ranking(query, n=N_CANDIDATES):
    records = bge_base_ranking(query, n)
    scores = get_reranker().predict([(query, d) for d, _ in records])
    order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    return [records[i] for i in order]


def resolve_homes(col):
    """Gold's home paper (model-independent fact about the corpus)."""
    data = col.get(include=["documents", "metadatas"])
    nd = [(_norm(d), m["source"]) for d, m in zip(data["documents"], data["metadatas"])]
    homes = []
    for case in EVAL_SET:
        g = _norm(case["gold"])
        srcs = sorted({s for t, s in nd if g in t})
        homes.append(srcs[0] if srcs else None)
    return homes


def _row(label, ranks):
    cells = "  ".join(f"{_recall_at_k(ranks)[k]:.2f}" for k in K_VALUES)
    return f"  {label:<16} {cells}  {_mrr(ranks):.3f}"


def main():
    global _new_col
    _new_col = build_index()
    homes = resolve_homes(_new_col)

    print("\nLoading cross-encoder reranker...")
    get_reranker()

    configs = [
        ("MiniLM base", minilm_base),
        ("MiniLM+rerank", minilm_reranked),
        ("BGE base", bge_base_ranking),
        ("BGE+rerank", bge_reranked_ranking),
    ]
    results = {label: eval_retriever(fn, homes) for label, fn in configs}

    header = "  " + " " * 16 + " " + "  ".join(f"R@{k}" for k in K_VALUES) + "    MRR"

    print("\n" + "=" * 72)
    print("EMBEDDING EXPERIMENT  --  all-MiniLM-L6-v2  vs  bge-small-en-v1.5")
    print(f"{len(EVAL_SET)} questions, candidate pool = {N_CANDIDATES}")
    print("=" * 72)

    print("\nCHUNK-LEVEL recall@k / MRR  (did we retrieve the exact gold passage?)")
    print("-" * 72)
    print(header)
    for label, _ in configs:
        print(_row(label, results[label][0]))
    print("-" * 72)

    print("\nPAPER-LEVEL recall@k / MRR  (was the right document found at all?)")
    print("-" * 72)
    print(header)
    for label, _ in configs:
        print(_row(label, results[label][1]))
    print("-" * 72)

    # Per-question chunk rank, to see which questions the stronger model rescued.
    print("\nPer-question rank of the gold chunk  ('-' = not in top 30)")
    print("-" * 72)
    print(f"  {'Q':<3} {'MiniLM':>6} {'MiniLM+rr':>10} {'BGE':>5} {'BGE+rr':>7}   question")
    cr = {label: results[label][0] for label, _ in configs}
    for i in range(len(EVAL_SET)):
        vals = [cr[label][i] for label, _ in configs]
        cells = [str(v) if v else "-" for v in vals]
        q = EVAL_SET[i]["question"]
        q = q if len(q) <= 40 else q[:37] + "..."
        print(f"  {i+1:<3} {cells[0]:>6} {cells[1]:>10} {cells[2]:>5} {cells[3]:>7}   {q}")
    print("-" * 72)


if __name__ == "__main__":
    main()

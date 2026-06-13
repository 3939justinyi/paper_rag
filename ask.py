import os

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"

# Retrieval backend. "bge" (default) uses BAAI/bge-small-en-v1.5, which roughly
# doubles retrieval recall over the original all-MiniLM-L6-v2 default and finds the
# right paper for ~9/10 eval questions (see retrieval_eval_results.md, section 4).
# Set RAG_EMBEDDING=minilm to fall back to the original index.
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING", "bge").lower()

claude = Anthropic()
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

if EMBEDDING_BACKEND == "minilm":
    # Original index: Chroma's built-in all-MiniLM-L6-v2 embeds the query for us.
    collection = chroma_client.get_collection("papers")
    _embed_query = None
else:
    # Stronger index: we embed the query ourselves, because BGE is asymmetric --
    # queries take an instruction prefix that passages do not.
    from sentence_transformers import SentenceTransformer

    _bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    collection = chroma_client.get_collection("papers_bge_small")

    def _embed_query(question):
        return _bge.encode([_BGE_QUERY_PREFIX + question], normalize_embeddings=True)[0].tolist()


def retrieve(question, n_results=5):
    if _embed_query is None:
        results = collection.query(query_texts=[question], n_results=n_results)
    else:
        results = collection.query(
            query_embeddings=[_embed_query(question)],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources


def answer(question, n_results=5):
    docs, sources = retrieve(question, n_results)
    context = "\n\n".join(f"[from {s}]\n{d}" for d, s in zip(docs, sources))
    prompt = (
        "Answer the question using only the context below, which is taken from "
        "research papers. Follow these rules:\n"
        "- Use only what the context states. If the answer is not in the context, "
        "say you don't know; do not fill gaps from general knowledge.\n"
        "- Do not add interpretation, framing, or implications the context does "
        "not state. Stick to the claims actually present.\n"
        "- Credit a finding to a paper only if the context shows that paper made "
        "it. Papers cite other studies; a reference appearing inside a paper is "
        "not that paper's own result, so do not attribute it to them.\n"
        "- Name the source papers you used.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, sources


if __name__ == "__main__":
    question = input("Ask a question about your papers: ")
    text, sources = answer(question)
    print("\n" + text)
    print("\nRetrieved from:", ", ".join(sorted(set(sources))))

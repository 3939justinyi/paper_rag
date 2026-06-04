import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers"

claude = Anthropic()
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)


def retrieve(question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources


def answer(question, n_results=5):
    docs, sources = retrieve(question, n_results)
    context = "\n\n".join(f"[from {s}]\n{d}" for d, s in zip(docs, sources))
    prompt = (
        "Answer the question using only the context below, which is taken from "
        "research papers. If the answer is not in the context, say you don't know. "
        "Name the source papers you used.\n\n"
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
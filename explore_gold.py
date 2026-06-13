"""Helper for hand-labeling gold snippets.

Searches the ACTUAL stored Chroma chunks (the same text retrieval returns) for
given terms within a given paper, and prints a window around each hit together
with the chunk id. Whatever phrase you copy from this output is guaranteed to
exist verbatim in a retrievable chunk, so substring matching in the eval will
behave the same way retrieval does.

Usage:
    venv/bin/python explore_gold.py "<source filename substring>" term1 term2 ...
"""
import re
import sys
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers"

collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)


def main():
    source_sub = sys.argv[1].lower()
    terms = sys.argv[2:]

    data = collection.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]
    ids = data["ids"]

    sources = sorted({m["source"] for m in metas})
    matched_sources = [s for s in sources if source_sub in s.lower()]
    print(f"# source substring {source_sub!r} -> {matched_sources}\n")

    for term in terms:
        print(f"================ TERM: {term!r} ================")
        pat = re.compile(re.escape(term), re.IGNORECASE)
        hits = 0
        for cid, doc, meta in zip(ids, docs, metas):
            if source_sub not in meta["source"].lower():
                continue
            for m in pat.finditer(doc):
                hits += 1
                a = max(0, m.start() - 120)
                b = min(len(doc), m.end() + 120)
                window = doc[a:b].replace("\n", " ")
                window = re.sub(r"\s+", " ", window).strip()
                print(f"[{cid} | {meta['source']}]")
                print(f"   ...{window}...")
                if hits >= 6:  # cap per term
                    break
            if hits >= 6:
                break
        if hits == 0:
            print("   (no hits)")
        print()


if __name__ == "__main__":
    main()

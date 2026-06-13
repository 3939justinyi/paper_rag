"""Retrieval evaluation for the paper_rag system.

This is separate from evaluate.py. evaluate.py grades *answer quality* (does the
generated answer capture the key facts). This file grades the two things that
sit underneath that answer:

  1. Retrieval quality   -- does the right passage actually get retrieved, and
                            how highly is it ranked? Measured with recall@k and
                            MRR, before and after a cross-encoder reranker.
  2. Faithfulness        -- is every claim in the generated answer supported by
                            the retrieved context (i.e. no hallucination)?
                            A separate judge call from the answer-quality judge.

Run:
    python retrieval_eval.py                 # gold check + retrieval before/after (no API calls)
    python retrieval_eval.py --faithfulness  # also run the faithfulness check (uses the API)
    python retrieval_eval.py --only-faithfulness
"""
import argparse
import re

import chromadb

# Faithfulness uses the live generation pipeline from the app (ask.retrieve/answer),
# so it tracks whatever retrieval backend ask.py is configured to use.
from ask import retrieve, answer, claude

# The retrieval-quality tables below always measure the original all-MiniLM-L6-v2
# baseline collection directly, so the documented "base" numbers stay fixed no
# matter which backend the app ships. (The MiniLM-vs-BGE comparison lives in
# embedding_experiment.py.)
collection = chromadb.PersistentClient(path="chroma_db").get_collection("papers")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
N_CANDIDATES = 30          # candidate pool pulled from Chroma before reranking
K_VALUES = [1, 3, 5, 10]   # recall@k cutoffs to report
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
JUDGE_MODEL = "claude-sonnet-4-6"   # same model family the app/judge already use

# ----------------------------------------------------------------------------
# Evaluation set
#
# These are the 10 existing eval questions (from evaluate.py). Each is paired
# with a GOLD snippet: a short, distinctive phrase copied verbatim from the single
# passage that answers it. A retrieved chunk counts as a hit when it contains the
# gold snippet (whitespace-normalized substring match -- see first_gold_rank).
#
# The gold phrases were lifted from the actual stored chunk text, so every one is
# guaranteed to live inside a retrievable chunk. Chunks are 1000 chars with 200
# overlap, so any phrase under ~200 chars always sits fully inside at least one
# chunk. >>> These are hand-labeled; confirm the phrases really are the answer-
# bearing words for each question and edit freely. <<<
# ----------------------------------------------------------------------------
EVAL_SET = [
    {
        "question": "What did Waddington originally mean by epigenetics, and how does today's definition differ?",
        "gold": "between genotype and phenotype lies a whole complex of development processes",
    },
    {
        "question": "What is DNA methylation, where does it occur, and what is its usual effect?",
        "gold": "predominantly found in cytosines of the dinucleotide sequence CpG",
    },
    {
        "question": "What is the epitranscriptome, and what did Widagdo and Bredy show about it?",
        "gold": "the level of m6A increases in the medial prefrontal cortex",
    },
    {
        "question": "How does neuronal activity change DNA methylation, and can methylation be removed?",
        "gold": "Gadd45b is required for activity-induced DNA demethylation of specific promoters",
    },
    {
        "question": "What is the evidence that epigenetic marks are required for memory?",
        "gold": "the memory suppressor gene PP1",
    },
    {
        "question": "How does maternal care program stress responses, and does it translate to humans?",
        "gold": "licking and grooming (LG) and arched-back nursing (ABN) by rat mothers altered the offspring epigenome",
    },
    {
        "question": "What chromatin changes are linked to depression and antidepressant action?",
        "gold": "viral-mediated HDAC5 overexpression in the hippocampus blocked",
    },
    {
        "question": "What role does neuroepigenetics play in addiction?",
        "gold": "in particular, in the mesolimbic dopamine system",
    },
    {
        "question": "What did Nugent 2015 reveal about how the brain becomes male or female?",
        "gold": "feminization must be actively maintained by DNA methylation",
    },
    {
        "question": "What is transgenerational epigenetic inheritance, and what is the landmark behavioral demonstration?",
        "gold": "CpG hypomethylation in the Olfr151 gene",
    },
]

# ----------------------------------------------------------------------------
# Matching: does a retrieved chunk contain the gold snippet?
# ----------------------------------------------------------------------------
def _norm(s):
    """Lowercase and collapse all whitespace, so PDF line breaks inside a chunk
    do not defeat the substring match."""
    return re.sub(r"\s+", " ", s).strip().lower()


def first_gold_rank(docs, gold):
    """1-based rank of the first retrieved chunk containing the gold snippet,
    or None if no retrieved chunk contains it."""
    g = _norm(gold)
    for i, doc in enumerate(docs, 1):
        if g in _norm(doc):
            return i
    return None


# ----------------------------------------------------------------------------
# Retrievers
#
# Each returns a ranking as a list of (chunk_text, source) records. Slicing the
# base ranking to k and keeping only chunk_text is identical to the app's
# retrieve(query, k), so it doubles as the baseline at every k. Carrying the
# source lets us also measure paper-level recall (see the diagnostic below).
# ----------------------------------------------------------------------------
def base_ranking(query, n=N_CANDIDATES):
    """Plain vector retrieval: the top-n chunks in Chroma's native distance order."""
    res = collection.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
    return list(zip(res["documents"][0], [m["source"] for m in res["metadatas"][0]]))


_reranker = None


def get_reranker():
    """Lazily load the cross-encoder (downloads ~80MB on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def reranked_ranking(query, n=N_CANDIDATES):
    """Pull n candidates from Chroma, score each (query, chunk) pair with the
    cross-encoder, and return them resorted by descending score. Reranking only
    reorders the pool, so it can never surface a chunk the base retriever did not
    already pull -- reranked recall@k is capped by base recall@n."""
    records = base_ranking(query, n)
    scores = get_reranker().predict([(query, doc) for doc, _ in records])
    order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    return [records[i] for i in order]


# ----------------------------------------------------------------------------
# Gold sanity check + home-paper resolution
# ----------------------------------------------------------------------------
def verify_gold():
    """Confirm every gold snippet exists in the indexed corpus, and resolve the
    home paper of each (the source of the chunk that contains it). Returns
    (all_present, homes) where homes[i] is the source filename for EVAL_SET[i]."""
    data = collection.get(include=["documents", "metadatas"])
    norm_docs = [(_norm(d), m["source"]) for d, m in zip(data["documents"], data["metadatas"])]

    print("Gold snippet check (is the answer passage even in the index?)")
    print("-" * 70)
    homes, all_ok = [], True
    for i, case in enumerate(EVAL_SET, 1):
        g = _norm(case["gold"])
        sources = sorted({src for nd, src in norm_docs if g in nd})
        homes.append(sources[0] if sources else None)
        all_ok &= bool(sources)
        flag = "ok " if sources else "!! MISSING"
        where = sources[0] if sources else "(no chunk contains this phrase)"
        print(f"  {flag} Q{i:<2} in {len(sources)} chunk(s): {where}")
    print("-" * 70)
    print("All gold snippets present.\n" if all_ok else "SOME GOLD MISSING -- fix the phrases above.\n")
    return all_ok, homes


# ----------------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------------
def _recall_at_k(ranks):
    n = len(ranks)
    return {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in K_VALUES}


def _mrr(ranks):
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def eval_retriever(rank_fn, homes):
    """Run a retriever over the eval set. rank_fn(question) -> list of
    (chunk, source) records. Returns chunk-level and paper-level rank lists.

    chunk-level: rank of the first chunk containing the gold snippet (strict --
                 the exact answer passage).
    paper-level: rank of the first chunk from the gold's home paper (lenient --
                 was the right document found at all)."""
    chunk_ranks, paper_ranks = [], []
    for case, home in zip(EVAL_SET, homes):
        records = rank_fn(case["question"])
        chunk_ranks.append(first_gold_rank([d for d, _ in records], case["gold"]))
        paper_ranks.append(next((i for i, (_, s) in enumerate(records, 1) if home and s == home), None))
    return chunk_ranks, paper_ranks


def _fmt_row(label, ranks):
    metrics = _recall_at_k(ranks)
    cells = "  ".join(f"{metrics[k]:.2f}" for k in K_VALUES)
    return f"  {label:<10} {cells}  {_mrr(ranks):.3f}"


def run_retrieval_eval():
    print("=" * 70)
    print("RETRIEVAL EVALUATION  (paper_rag)")
    print(f"candidate pool = {N_CANDIDATES} chunks per query, {len(EVAL_SET)} questions")
    print("=" * 70 + "\n")

    _, homes = verify_gold()

    base_chunk, base_paper = eval_retriever(lambda q: base_ranking(q, N_CANDIDATES), homes)

    print("Loading cross-encoder reranker (first run downloads the model)...")
    get_reranker()
    rr_chunk, rr_paper = eval_retriever(lambda q: reranked_ranking(q, N_CANDIDATES), homes)

    header = "             " + "  ".join(f"R@{k}" for k in K_VALUES) + "    MRR"

    # ---- Primary result: chunk-level (the spec's metric) ----
    print("\nPRIMARY RESULT -- chunk-level recall@k and MRR")
    print("(a hit means the exact gold answer-passage was retrieved)")
    print("-" * 70)
    print(header)
    print(_fmt_row("base", base_chunk))
    print(_fmt_row("reranked", rr_chunk))
    print("-" * 70)

    # ---- Per-question movement ----
    print("\nPer-question rank of the gold chunk  ('-' = not in pool)")
    print("-" * 70)
    print(f"  {'Q':<3} {'base':>5} {'rerank':>7}   question")
    for i, (b, r) in enumerate(zip(base_chunk, rr_chunk), 1):
        q = EVAL_SET[i - 1]["question"]
        q = q if len(q) <= 50 else q[:47] + "..."
        print(f"  {i:<3} {str(b) if b else '-':>5} {str(r) if r else '-':>7}   {q}")
    print("-" * 70)

    # ---- Diagnostic: paper-level, to read the baseline correctly ----
    print("\nDIAGNOSTIC -- paper-level recall@k (was the right *document* found at all?)")
    print("(lenient: a hit means any chunk from the gold's home paper was retrieved)")
    print("-" * 70)
    print(header)
    print(_fmt_row("base", base_paper))
    print(_fmt_row("reranked", rr_paper))
    print("-" * 70)

    _diagnose(base_chunk, base_paper)
    return base_chunk, base_paper, rr_chunk, rr_paper


def _diagnose(chunk_ranks, paper_ranks):
    """Separate the two failure modes the spec warns about."""
    kmax = max(K_VALUES)
    chunk_top = _recall_at_k(chunk_ranks)[kmax]
    paper_top = _recall_at_k(paper_ranks)[kmax]

    true_miss = sum(1 for p in paper_ranks if p is None or p > kmax)
    wrong_chunk = sum(1 for c, p in zip(chunk_ranks, paper_ranks)
                      if (c is None or c > kmax) and p is not None and p <= kmax)

    print("\nReading the baseline (before changing anything):")
    print(f"  chunk-level recall@{kmax} = {chunk_top:.2f}   paper-level recall@{kmax} = {paper_top:.2f}")
    print(f"  - {true_miss}/{len(EVAL_SET)} questions: home paper NOT in top {kmax} -> a true retrieval")
    print("    miss (embedding/chunking/depth). A reranker only reorders the pool, so it")
    print("    cannot recover these.")
    print(f"  - {wrong_chunk}/{len(EVAL_SET)} questions: home paper IS in top {kmax} but the exact gold")
    print("    chunk is not -> a ranking issue within the right document, which is the")
    print("    case a cross-encoder reranker is meant to improve.")


# ----------------------------------------------------------------------------
# Faithfulness check  (separate from the answer-quality judge in evaluate.py)
# ----------------------------------------------------------------------------
def _build_context(question, n_results=5):
    """Rebuild the exact context string answer() fed to the model for this
    question (same retrieval, same formatting as ask.answer)."""
    docs, sources = retrieve(question, n_results)
    return "\n\n".join(f"[from {s}]\n{d}" for d, s in zip(docs, sources))


def faithfulness_judge(context, candidate):
    """One judge call asking whether every claim in the answer is supported by the
    context. Returns (is_faithful, raw verdict text)."""
    prompt = (
        "You are checking an answer for faithfulness to its source context, NOT "
        "for whether it is a good answer. Read the context and the answer. Decide "
        "whether EVERY factual claim in the answer is directly supported by the "
        "context. A claim that is true in general but not present in the context "
        "counts as unsupported.\n\n"
        "Reply with exactly FAITHFUL or UNFAITHFUL on the first line. If "
        "UNFAITHFUL, list each unsupported claim on its own line after that.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{candidate}"
    )
    response = claude.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    return text.upper().startswith("FAITHFUL"), text


def run_faithfulness_eval():
    print("\n" + "=" * 70)
    print("FAITHFULNESS CHECK  (is every claim in the answer grounded in context?)")
    print("=" * 70 + "\n")

    faithful, unfaithful = 0, []
    for i, case in enumerate(EVAL_SET, 1):
        q = case["question"]
        candidate, _ = answer(q)                 # the real pipeline's answer
        context = _build_context(q)              # the context that answer used
        ok, verdict = faithfulness_judge(context, candidate)
        if ok:
            faithful += 1
            print(f"  Q{i:<2} FAITHFUL")
        else:
            unfaithful.append((i, q, candidate, verdict))
            print(f"  Q{i:<2} UNFAITHFUL")

    n = len(EVAL_SET)
    print("\n" + "-" * 70)
    print(f"Faithful answers: {faithful}/{n}  ({faithful / n:.0%})")
    print("-" * 70)

    if unfaithful:
        print("\nAnswers with unsupported claims (examples to point to):")
        for i, q, candidate, verdict in unfaithful:
            print("\n" + "=" * 70)
            print(f"Q{i}: {q}")
            print("-" * 70)
            print("Judge verdict:")
            print(verdict)
            print("-" * 70)
            print("Answer that was graded:")
            print(candidate)
    return faithful, unfaithful


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval + faithfulness evaluation for paper_rag")
    parser.add_argument("--faithfulness", action="store_true",
                        help="also run the faithfulness check (makes API calls)")
    parser.add_argument("--only-faithfulness", action="store_true",
                        help="run only the faithfulness check")
    args = parser.parse_args()

    if not args.only_faithfulness:
        run_retrieval_eval()
    if args.faithfulness or args.only_faithfulness:
        run_faithfulness_eval()

# Paper RAG

A retrieval augmented generation (RAG) app that answers questions about a collection of research papers and cites the papers it used. You point it at a folder of PDFs, it builds a searchable index, and you ask questions in plain English through a web interface or the command line.

## What it does

The papers are too many to paste into a language model at once, so the app retrieves only the passages relevant to each question and hands those to the model to write a grounded answer. Every answer names the source papers it drew from, and the model is told to say when the context does not contain the answer rather than guess.

## How it works

1. Ingestion. Each PDF is read and split into overlapping text chunks.
2. Indexing. Every chunk is embedded into a vector and stored in a local Chroma database, so the embedding work happens once.
3. Retrieval. A question is embedded the same way, and Chroma returns the chunks closest to it in meaning.
4. Generation. The retrieved chunks and the question go to Claude, which answers using only that context and names the sources.

## Tech stack

- Python
- pypdf for text extraction
- Chroma as the vector store, with BAAI/bge-small-en-v1.5 embeddings by default (set `RAG_EMBEDDING=minilm` to fall back to the original built-in all-MiniLM-L6-v2 index)
- Anthropic Claude for answer generation
- Streamlit for the web interface

## Setup

Create a virtual environment and install the dependencies.

```
python3 -m venv venv
source venv/bin/activate
pip install chromadb anthropic pypdf streamlit python-dotenv sentence-transformers
```

Add your Anthropic API key to a file called .env in the project root.

```
ANTHROPIC_API_KEY=your_key_here
```

Put your PDFs in a folder called data, then build the indexes. This builds both the default `bge-small-en-v1.5` index and the original `all-MiniLM-L6-v2` index (used by the `RAG_EMBEDDING=minilm` fallback and the evaluations), so the first run downloads the embedding model and may take a few minutes.

```
python index.py
```

## Running it

Command line.

```
python ask.py
```

Web interface.

```
python -m streamlit run app.py
```

## Evaluation

The project includes an evaluation harness that measures answer quality with an LLM as judge. A set of questions with reference answers is run through the app, and a separate model call grades whether each answer captures the key facts, producing a score.

```
python evaluate.py
```

On a test set of 10 questions over 24 neuroscience papers, the system scored 7 out of 10. I then ran two tuning experiments. Retrieving more chunks per question dropped the score to 6, because the extra context diluted the relevant passage. Using smaller chunks also landed at 6, fixing one question while breaking another. The same three questions failed across every configuration, which showed the remaining errors were not a tuning problem but a limit of the small embedding model and the basic retrieval approach. Closing those gaps would need reranking the retrieved results or a stronger embedding model.

### Retrieval and faithfulness evaluation

`retrieval_eval.py` measures retrieval directly (recall@k and MRR against hand-labeled gold passages, before and after a cross-encoder reranker) and checks faithfulness (whether every claim in an answer is grounded in the retrieved context). `embedding_experiment.py` compares the embedding models. Full tables are in `retrieval_eval_results.md`. Headline findings, on a corpus that has since grown to 42 papers:

- Retrieval, not ranking, was the bottleneck: the right paper was often not retrieved at all, which a reranker cannot fix.
- Swapping the default embeddings to `bge-small-en-v1.5` roughly doubled retrieval recall (paper-level recall@10 0.60 to 0.90) and lifted answer quality from 5/10 to 7/10.
- A cross-encoder reranker helps the weak MiniLM retriever but hurts the stronger BGE one, so it is not used in the app.
- Tightening the answer prompt raised faithfulness from 8/10 to 10/10 with no quality cost.

```
python retrieval_eval.py                 # retrieval recall/MRR, before vs after reranking
python retrieval_eval.py --faithfulness  # add the faithfulness check (uses the API)
python embedding_experiment.py           # MiniLM vs BGE embedding comparison
```

## Limitations and next steps

The system handles broad conceptual questions well but can still miss multi-paper questions, where one of two needed papers is not retrieved. The reranking and stronger-embedding ideas below were tested (see above); the embedding upgrade is now the default. The clearest remaining improvements would be a larger or domain-matched embedding model (e.g. `bge-base`), query rewriting for definitional questions, and more precise passage level citations.
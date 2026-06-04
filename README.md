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
- Chroma as the vector store, with its built in all-MiniLM-L6-v2 embeddings
- Anthropic Claude for answer generation
- Streamlit for the web interface

## Setup

Create a virtual environment and install the dependencies.

```
python3 -m venv venv
source venv/bin/activate
pip install chromadb anthropic pypdf streamlit python-dotenv
```

Add your Anthropic API key to a file called .env in the project root.

```
ANTHROPIC_API_KEY=your_key_here
```

Put your PDFs in a folder called data, then build the index.

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

## Limitations and next steps

The system handles broad conceptual questions well but can miss narrow details that sit in a single passage, since the embedding model does not always rank that passage highly. The clearest improvements would be a reranking step over the retrieved chunks and a stronger embedding model, then more precise passage level citations.
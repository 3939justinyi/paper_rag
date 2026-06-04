from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def load_chunks(data_dir="data", chunk_size=1000, overlap=200):
    chunks = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        text = extract_text(pdf_path)
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({
                "id": f"{pdf_path.stem}_{i}",
                "text": chunk,
                "source": pdf_path.name,
            })
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from the PDFs in data/")
    if chunks:
        print("\nExample chunk:")
        print("source:", chunks[0]["source"])
        print(chunks[0]["text"][:300])
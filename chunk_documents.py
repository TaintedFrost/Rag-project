import pymupdf
from pathlib import Path


DATA_FOLDER = Path("data")


def extract_pages(pdf_path):
    """Extract text from each page of a PDF."""
    
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text().strip()

        if text:
            pages.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_number + 1
            })

    document.close()

    return pages


def create_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


all_chunks = []

pdf_files = [
    DATA_FOLDER / "doc1.pdf",
    DATA_FOLDER / "doc2.pdf",
    DATA_FOLDER / "doc3.pdf",
    DATA_FOLDER / "doc4.pdf",
]


for pdf_path in pdf_files:

    print(f"Processing: {pdf_path.name}")

    pages = extract_pages(pdf_path)

    for page in pages:

        chunks = create_chunks(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })


print()
print(f"Total chunks created: {len(all_chunks)}")

print("\nFirst 3 chunks:\n")

for i, chunk in enumerate(all_chunks[:3]):

    print("=" * 70)
    print(f"Chunk {i + 1}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print("=" * 70)
    print(chunk["text"])
    print()
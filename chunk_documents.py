import json

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
    """Split text into chunks while trying to preserve paragraph boundaries."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding this paragraph stays within the limit,
        # keep building the current chunk.
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += "\n"

            current_chunk += paragraph

        else:

            # Save the current chunk
            if current_chunk:
                chunks.append(current_chunk)

            # Start a new chunk
            current_chunk = paragraph

    # Save the final chunk
    if current_chunk:
        chunks.append(current_chunk)

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


output_path = Path("chunks.json")

with output_path.open("w", encoding="utf-8") as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print(f"Saved chunks to: {output_path}")


print("\nFirst 3 chunks:\n")

for i, chunk in enumerate(all_chunks[:3]):

    print("=" * 70)
    print(f"Chunk {i + 1}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print("=" * 70)
    print(chunk["text"])
    print()
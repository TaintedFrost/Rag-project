import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

METADATA_PATH = "chunk_metadata.json"

INDEX_FOLDER = Path("indexes")


# ============================================================
# Document categories
# ============================================================

DOCUMENT_CATEGORIES = {
    "doc1.pdf": "doctorat",
    "doc2.pdf": "licenta",
    "doc3.pdf": "burse",
    "doc4.pdf": "masterat",
}


# ============================================================
# Load chunks
# ============================================================

print("Loading chunks...")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# Create output folder
# ============================================================

INDEX_FOLDER.mkdir(exist_ok=True)


# ============================================================
# Load embedding model
# ============================================================

print()
print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# Create an index for each category
# ============================================================

categories = {}

for chunk_index, chunk in enumerate(chunks):

    source = chunk["source"]

    category = DOCUMENT_CATEGORIES.get(source)

    if category is None:
        continue

    if category not in categories:
        categories[category] = []

    categories[category].append(chunk_index)


for category, chunk_indices in categories.items():

    print()
    print("=" * 70)
    print(f"Creating index: {category}")
    print(f"Chunks: {len(chunk_indices)}")
    print("=" * 70)

    category_chunks = [
        chunks[i]
        for i in chunk_indices
    ]

    texts = [
        chunk["text"]
        for chunk in category_chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    index_path = INDEX_FOLDER / f"{category}.index"

    faiss.write_index(
        index,
        str(index_path)
    )


    # Save the original chunk indices so that we know
    # which chunk each vector corresponds to.
    mapping_path = (
        INDEX_FOLDER / f"{category}_mapping.json"
    )

    with mapping_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunk_indices,
            file,
            indent=2
        )


    print(
        f"Saved index: {index_path}"
    )

    print(
        f"Saved mapping: {mapping_path}"
    )


print()
print("=" * 70)
print("All category indexes created.")
print("=" * 70)
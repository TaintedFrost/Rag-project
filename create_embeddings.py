import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load our chunks
# --------------------------------------------------

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# 2. Load the embedding model
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 3. Extract the text from every chunk
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 5. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")


# --------------------------------------------------
# 6. Save the index
# --------------------------------------------------

faiss.write_index(index, "faiss_index.bin")

print("Saved FAISS index to faiss_index.bin")


# --------------------------------------------------
# 7. Save metadata
# --------------------------------------------------

with open("chunk_metadata.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Saved chunk metadata to chunk_metadata.json")

print("\nDone!")
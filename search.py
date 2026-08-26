import json
import faiss

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Load the saved data
# --------------------------------------------------

print("Loading data...")

index = faiss.read_index("faiss_index.bin")

with open("chunk_metadata.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# --------------------------------------------------
# Load the same embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# Search function
# --------------------------------------------------

def search(query, top_k=5):

    # Convert the question into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # FAISS expects float32
    query_embedding = query_embedding.astype("float32")

    # Search the vector database
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        chunk = chunks[index_number]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"]
        })

    return results


# --------------------------------------------------
# Interactive search
# --------------------------------------------------

print("\nRAG document search")
print("Type 'exit' to stop.\n")


while True:

    query = input("Întrebare: ")

    if query.lower() == "exit":
        break

    results = search(query)

    print("\n" + "=" * 80)
    print("REZULTATE")
    print("=" * 80)

    for i, result in enumerate(results):

        print(f"\n--- Rezultat {i + 1} ---")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Sursă: {result['source']}")
        print(f"Pagina: {result['page']}")
        print()
        print(result["text"])

    print()
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunk_metadata.json"

# We retrieve many candidates first.
INITIAL_RESULTS = 50

# Number of results shown to the user.
FINAL_RESULTS = 5


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
# Keywords associated with each category
# ============================================================

CATEGORY_KEYWORDS = {

    "doctorat": [
        "doctorat",
        "doctor",
        "doctorand",
        "doctoranzi",
        "studii doctorale",
        "școala doctorală",
        "școlile doctorale",
        "candidat doctorat",
    ],

    "licenta": [
        "licență",
        "licenta",
        "licențiat",
        "licență",
        "studii de licență",
        "ciclu de licență",
        "admitere licență",
        "admiterea la licență",
    ],

    "burse": [
        "bursă",
        "burse",
        "bursei",
        "burselor",
        "bursa",
        "bursă de performanță",
        "bursă de merit",
        "bursă socială",
        "ajutor social",
    ],

    "masterat": [
        "masterat",
        "master",
        "masterand",
        "studii de masterat",
        "studii de master",
        "ciclu de masterat",
        "admitere masterat",
        "admiterea la masterat",
    ],
}


# ============================================================
# Load FAISS and metadata
# ============================================================

print("Loading data...")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# Load embedding model
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Lowercase the text and return its words.
    Romanian diacritics are preserved.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


# ============================================================
# Detect likely document category
# ============================================================

def detect_categories(query):
    """
    Determine which document categories are relevant
    based on important words in the user's question.
    """

    query_lower = query.lower()

    detected = []

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in query_lower:
                detected.append(category)
                break

    return detected


# ============================================================
# Keyword score
# ============================================================

def keyword_score(query, text):
    """
    Calculate the proportion of query words that
    also appear in the retrieved text.
    """

    query_words = set(normalize_text(query))
    text_words = set(normalize_text(text))

    if not query_words:
        return 0.0

    overlap = query_words.intersection(text_words)

    return len(overlap) / len(query_words)


# ============================================================
# Category relevance
# ============================================================

def category_score(source, detected_categories):
    """
    Give a large bonus when a retrieved chunk comes
    from the document category implied by the question.
    """

    if not detected_categories:
        return 0.0

    category = DOCUMENT_CATEGORIES.get(source)

    if category in detected_categories:
        return 1.0

    return 0.0


# ============================================================
# Search
# ============================================================

def search(query, top_k=FINAL_RESULTS):

    # --------------------------------------------------------
    # 1. Detect topic
    # --------------------------------------------------------

    detected_categories = detect_categories(query)

    if detected_categories:
        print(
            f"\nDetected topic: "
            f"{', '.join(detected_categories)}"
        )
    else:
        print("\nDetected topic: general")


    # --------------------------------------------------------
    # 2. Convert query to embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # 3. Retrieve many candidates
    # --------------------------------------------------------

    semantic_scores, indices = index.search(
        query_embedding,
        INITIAL_RESULTS
    )


    candidates = []


    # --------------------------------------------------------
    # 4. Score candidates
    # --------------------------------------------------------

    for semantic_score, index_number in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        chunk = chunks[index_number]

        source = chunk["source"]

        lexical_score = keyword_score(
            query,
            chunk["text"]
        )

        topic_score = category_score(
            source,
            detected_categories
        )


        # ----------------------------------------------------
        # Combined score
        #
        # Semantic similarity:
        # 60%
        #
        # Keyword overlap:
        # 15%
        #
        # Correct document category:
        # 25%
        # ----------------------------------------------------

        combined_score = (
            0.60 * float(semantic_score)
            +
            0.15 * lexical_score
            +
            0.25 * topic_score
        )


        candidates.append({
            "semantic_score": float(semantic_score),
            "keyword_score": float(lexical_score),
            "topic_score": float(topic_score),
            "combined_score": combined_score,
            "text": chunk["text"],
            "source": source,
            "page": chunk["page"],
        })


    # --------------------------------------------------------
    # 5. If a specific topic was detected, prioritize only
    #    chunks from that document category.
    # --------------------------------------------------------

    if detected_categories:

        topic_candidates = [
            candidate
            for candidate in candidates
            if DOCUMENT_CATEGORIES.get(candidate["source"])
            in detected_categories
        ]

        # If we found candidates in the correct document,
        # use those instead of unrelated documents.
        if topic_candidates:
            candidates = topic_candidates


    # --------------------------------------------------------
    # 6. Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )


    return candidates[:top_k]


# ============================================================
# Interactive interface
# ============================================================

print()
print("Romanian RAG document search")
print("Type 'exit' to stop.")
print()


while True:

    try:

        query = input("Întrebare: ").strip()

    except (KeyboardInterrupt, EOFError):

        print("\nExiting...")
        break


    if query.lower() == "exit":
        break


    if not query:
        continue


    results = search(query)


    print()
    print("=" * 80)
    print("REZULTATE")
    print("=" * 80)


    if not results:

        print("\nNu au fost găsite rezultate relevante.")
        print()

        continue


    for i, result in enumerate(results, start=1):

        print()
        print(f"--- Rezultat {i} ---")

        print(
            f"Semantic similarity: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Keyword score:       "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"Topic score:         "
            f"{result['topic_score']:.4f}"
        )

        print(
            f"Combined score:      "
            f"{result['combined_score']:.4f}"
        )

        print(
            f"Sursă:               "
            f"{result['source']}"
        )

        print(
            f"Pagina:              "
            f"{result['page']}"
        )

        print()
        print(result["text"])

    print()
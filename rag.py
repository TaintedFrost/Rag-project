import json
import re

import faiss
import numpy as np
from ollama import chat
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_NAME = "qwen3:4b"

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunk_metadata.json"

# Retrieve more candidates initially.
INITIAL_RESULTS = 50

# Number of chunks that will actually be sent to the LLM.
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
# Keywords used to detect the topic of a question
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
        "studii de licență",
        "ciclu de licență",
        "admitere licență",
        "admiterea la licență",
    ],

    "burse": [
        "bursă",
        "bursa",
        "burse",
        "bursei",
        "burselor",
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
# Load FAISS index and metadata
# ============================================================

print("Loading RAG data...")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# Load embedding model
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

embedding_model = SentenceTransformer(MODEL_NAME)


# ============================================================
# Text helpers
# ============================================================

def normalize_text(text):
    """
    Lowercase text and extract words.
    Romanian diacritics are preserved.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def keyword_score(query, text):
    """
    Measure how many unique query words also occur in the text.
    """

    query_words = set(normalize_text(query))
    text_words = set(normalize_text(text))

    if not query_words:
        return 0.0

    overlap = query_words.intersection(text_words)

    return len(overlap) / len(query_words)


# ============================================================
# Detect document category
# ============================================================

def detect_categories(query):
    """
    Detect which document type is most likely relevant
    to the user's question.
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
# Calculate topic score
# ============================================================

def topic_score(source, detected_categories):

    if not detected_categories:
        return 0.0

    source_category = DOCUMENT_CATEGORIES.get(source)

    if source_category in detected_categories:
        return 1.0

    return 0.0


# ============================================================
# Retrieve relevant chunks
# ============================================================

def retrieve(query, top_k=FINAL_RESULTS):

    detected_categories = detect_categories(query)

    # --------------------------------------------------------
    # Create embedding for the question
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    semantic_scores, indices = index.search(
        query_embedding,
        INITIAL_RESULTS
    )


    candidates = []


    # --------------------------------------------------------
    # Combine semantic + lexical + topic scores
    # --------------------------------------------------------

    for semantic_similarity, index_number in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        chunk = chunks[index_number]

        source = chunk["source"]

        lexical = keyword_score(
            query,
            chunk["text"]
        )

        topic = topic_score(
            source,
            detected_categories
        )

        combined = (
            0.60 * float(semantic_similarity)
            +
            0.15 * lexical
            +
            0.25 * topic
        )

        candidates.append({
            "text": chunk["text"],
            "source": source,
            "page": chunk["page"],
            "semantic_score": float(semantic_similarity),
            "keyword_score": lexical,
            "topic_score": topic,
            "combined_score": combined,
        })


    # --------------------------------------------------------
    # If a specific category was detected, prioritize chunks
    # belonging to that category.
    # --------------------------------------------------------

    if detected_categories:

        category_candidates = [
            candidate
            for candidate in candidates
            if DOCUMENT_CATEGORIES.get(candidate["source"])
            in detected_categories
        ]

        if category_candidates:
            candidates = category_candidates


    # --------------------------------------------------------
    # Sort by combined score
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return candidates[:top_k], detected_categories


# ============================================================
# Build the prompt for the LLM
# ============================================================

def build_prompt(query, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
CONTEXT {i}
Sursă: {chunk['source']}
Pagina: {chunk['page']}

{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
Ești un asistent care răspunde la întrebări despre
regulamentele Universității Naționale de Știință și
Tehnologie POLITEHNICA București.

Răspunde ÎN LIMBA ROMÂNĂ.

Folosește DOAR informațiile din CONTEXT pentru a răspunde.

Nu inventa informații.

Dacă răspunsul nu poate fi determinat din context,
spune clar:

"Nu am găsit această informație în documentele disponibile."

Răspunsul trebuie să fie clar și concis.

La final, menționează sursa/sursele folosite în forma:

Surse:
- document.pdf, pagina X


ÎNTREBARE:
{query}


CONTEXT:
{context}
"""

    return prompt


# ============================================================
# Generate answer with Qwen
# ============================================================

def generate_answer(query, retrieved_chunks):

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    response = chat(
        model=LLM_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# ============================================================
# Interactive RAG application
# ============================================================

print()
print("=" * 70)
print("ROMANIAN RAG ASSISTANT")
print("=" * 70)
print("Model:", LLM_NAME)
print("Type 'exit' to stop.")
print()


while True:

    try:
        query = input("Întrebare: ").strip()

    except (KeyboardInterrupt, EOFError):

        print("\nExiting...")
        break


    if query.lower() == "exit":
        print("Exiting...")
        break


    if not query:
        continue


    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retrieved_chunks, detected_categories = retrieve(query)


    print()

    if detected_categories:
        print(
            "Topic detectat:",
            ", ".join(detected_categories)
        )
    else:
        print("Topic detectat: general")


    print("\nFragmente recuperate:")

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"{i}. {chunk['source']}, "
            f"pagina {chunk['page']} "
            f"(scor {chunk['combined_score']:.4f})"
        )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print("\nSe generează răspunsul...\n")

    try:

        answer = generate_answer(
            query,
            retrieved_chunks
        )

        print("=" * 70)
        print("RĂSPUNS")
        print("=" * 70)

        print(answer)

    except Exception as error:

        print("=" * 70)
        print("EROARE LLM")
        print("=" * 70)

        print(error)

    print()
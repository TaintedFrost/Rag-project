import json
import re
import time
from pathlib import Path

import faiss
import numpy as np

from ollama import chat
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LLM_NAME = "qwen3:1.7b"

METADATA_PATH = "chunk_metadata.json"
INDEX_FOLDER = Path("indexes")

# Number of candidates retrieved by each query.
SEARCH_RESULTS = 10

# Number of chunks sent to the LLM.
FINAL_RESULTS = 5

# Maximum generated tokens.
MAX_GENERATED_TOKENS = 180


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
# Category keywords
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
        "admitere la doctorat",
        "înscrierea la doctorat",
    ],

    "licenta": [
        "licență",
        "licenta",
        "studii de licență",
        "ciclu de licență",
        "admitere licență",
        "admiterea la licență",
        "concursul de admitere la licență",
    ],

    "burse": [
        "bursă",
        "bursa",
        "burse",
        "bursei",
        "burselor",
        "bursă de performanță",
        "bursa de performanță",
        "bursă de merit",
        "bursa de merit",
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
        "concursul de admitere la masterat",
    ],
}


# ============================================================
# Additional semantic search phrases
#
# IMPORTANT:
# These are searched separately from the user's question.
# We do NOT concatenate them into one artificial query.
# ============================================================

CATEGORY_SEARCH_PHRASES = {

    "doctorat": [
        "condiții înscriere la doctorat",
        "dosarul de înscriere la doctorat",
    ],

    "licenta": [
        "condiții participare concurs admitere licență",
        "organizarea concursului de admitere licență",
    ],

    "burse": [
        "criterii pentru acordarea bursei de performanță",
        "condiții bursa de performanță și punctaj",
    ],

    "masterat": [
        "condiții admitere studii universitare de masterat",
        "organizarea concursului de admitere masterat",
    ],
}


# ============================================================
# Load chunks
# ============================================================

print("Loading RAG data...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)

print(
    f"Loaded {len(chunks)} total chunks."
)


# ============================================================
# Load embedding model
# ============================================================

print(
    "Loading embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

model_load_start = time.perf_counter()

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

model_load_time = (
    time.perf_counter()
    - model_load_start
)

print(
    f"Embedding model loaded in "
    f"{model_load_time:.2f} seconds."
)


# ============================================================
# Load category indexes
# ============================================================

category_indexes = {}
category_mappings = {}

for category in set(
    DOCUMENT_CATEGORIES.values()
):

    index_path = (
        INDEX_FOLDER /
        f"{category}.index"
    )

    mapping_path = (
        INDEX_FOLDER /
        f"{category}_mapping.json"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing index: {index_path}\n"
            f"Run create_category_indexes.py first."
        )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing mapping: {mapping_path}\n"
            f"Run create_category_indexes.py first."
        )

    category_indexes[category] = (
        faiss.read_index(
            str(index_path)
        )
    )

    with mapping_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        category_mappings[category] = (
            json.load(file)
        )


# ============================================================
# Text utilities
# ============================================================

def normalize_text(text):

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def keyword_score(query, text):

    query_words = set(
        normalize_text(query)
    )

    text_words = set(
        normalize_text(text)
    )

    if not query_words:
        return 0.0

    overlap = (
        query_words.intersection(
            text_words
        )
    )

    return (
        len(overlap)
        /
        len(query_words)
    )


# ============================================================
# Category detection
# ============================================================

def detect_category(query):

    query_lower = query.lower()

    for category, keywords in (
        CATEGORY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in query_lower:
                return category

    return None


# ============================================================
# Retrieve using one query
# ============================================================

def semantic_search(
    query,
    category
):

    index = category_indexes[
        category
    ]

    mapping = category_mappings[
        category
    ]

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    search_k = min(
        SEARCH_RESULTS,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    results = []

    for score, local_index in zip(
        scores[0],
        indices[0]
    ):

        if local_index < 0:
            continue

        original_index = mapping[
            local_index
        ]

        chunk = chunks[
            original_index
        ]

        results.append({
            "global_index": original_index,
            "semantic_score": float(
                score
            ),
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
        })

    return results


# ============================================================
# Multi-query retrieval
# ============================================================

def retrieve(query):

    retrieval_start = time.perf_counter()

    category = detect_category(
        query
    )

    if category is None:

        # Fallback to global FAISS index.

        index = faiss.read_index(
            "faiss_index.bin"
        )

        query_embedding = (
            embedding_model.encode(
                [query],
                normalize_embeddings=True
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        embedding_time_start = (
            time.perf_counter()
        )

        scores, indices = index.search(
            query_embedding,
            FINAL_RESULTS
        )

        embedding_time = (
            time.perf_counter()
            - embedding_time_start
        )

        results = []

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number < 0:
                continue

            chunk = chunks[
                index_number
            ]

            results.append({
                "global_index": index_number,
                "semantic_score": float(
                    score
                ),
                "keyword_score": keyword_score(
                    query,
                    chunk["text"]
                ),
                "combined_score": float(
                    score
                ),
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
            })

        timing = {
            "embedding": embedding_time,
            "search": 0.0,
            "reranking": 0.0,
            "total_retrieval": (
                time.perf_counter()
                -
                retrieval_start
            ),
        }

        return (
            results,
            category,
            timing
        )


    # ========================================================
    # Category-specific retrieval
    # ========================================================

    queries = [
        query
    ]

    queries.extend(
        CATEGORY_SEARCH_PHRASES.get(
            category,
            []
        )
    )


    # --------------------------------------------------------
    # Search every query separately.
    # --------------------------------------------------------

    all_results = {}

    embedding_start = time.perf_counter()

    for search_query in queries:

        results = semantic_search(
            search_query,
            category
        )

        for result in results:

            index_number = (
                result["global_index"]
            )

            if index_number not in all_results:

                all_results[
                    index_number
                ] = {
                    "global_index": index_number,
                    "text": result["text"],
                    "source": result["source"],
                    "page": result["page"],
                    "semantic_scores": [],
                }

            all_results[
                index_number
            ]["semantic_scores"].append(
                result["semantic_score"]
            )


    embedding_time = (
        time.perf_counter()
        -
        embedding_start
    )


    # --------------------------------------------------------
    # Combine scores.
    #
    # We use the BEST semantic score from the
    # original question or auxiliary query.
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    candidates = []


    for candidate in all_results.values():

        best_semantic = max(
            candidate["semantic_scores"]
        )

        lexical = keyword_score(
            query,
            candidate["text"]
        )

        combined = (
            0.85 * best_semantic
            +
            0.15 * lexical
        )

        candidates.append({
            "global_index": (
                candidate["global_index"]
            ),
            "semantic_score": best_semantic,
            "keyword_score": lexical,
            "combined_score": combined,
            "text": candidate["text"],
            "source": candidate["source"],
            "page": candidate["page"],
        })


    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Remove duplicate pages.
    # --------------------------------------------------------

    selected = []

    seen_pages = set()

    for candidate in candidates:

        page_key = (
            candidate["source"],
            candidate["page"]
        )

        if page_key in seen_pages:
            continue

        seen_pages.add(
            page_key
        )

        selected.append(
            candidate
        )

        if len(selected) >= FINAL_RESULTS:
            break


    reranking_time = (
        time.perf_counter()
        -
        rerank_start
    )

    total_retrieval = (
        time.perf_counter()
        -
        retrieval_start
    )

    timing = {
        "embedding": embedding_time,
        "search": 0.0,
        "reranking": reranking_time,
        "total_retrieval": total_retrieval,
    }

    return (
        selected,
        category,
        timing
    )


# ============================================================
# Build prompt
# ============================================================

def build_prompt(
    query,
    retrieved_chunks
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
CONTEXT {i}
Sursa: {chunk['source']}
Pagina: {chunk['page']}

{chunk['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    return f"""
/no_think

Răspunde la întrebarea utilizatorului folosind
EXCLUSIV informațiile din CONTEXT.

Întrebare:
{query}

Reguli:
- Răspunde în limba română.
- Răspunde direct.
- Include toate condițiile relevante care apar în context.
- Păstrează valorile numerice exacte.
- Include ani de studiu, medii, punctaje,
  perioade și limite atunci când sunt relevante.
- Nu inventa informații.
- Nu confunda tipurile de burse.
- Nu adăuga informații generale.
- Dacă informația nu poate fi găsită în context, spune:
"Nu am găsit această informație în documentele disponibile."

Surse:
- document.pdf, pagina X

CONTEXT:
{context}

Răspuns:
"""


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    query,
    retrieved_chunks
):

    llm_start = time.perf_counter()

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
        ],
        think=False,
        options={
            "temperature": 0.1,
            "num_predict": MAX_GENERATED_TOKENS,
        }
    )

    llm_time = (
        time.perf_counter()
        -
        llm_start
    )

    answer = response.message.content

    prompt_tokens = getattr(
        response,
        "prompt_eval_count",
        None
    )

    generated_tokens = getattr(
        response,
        "eval_count",
        None
    )

    return (
        answer,
        llm_time,
        prompt_tokens,
        generated_tokens
    )


# ============================================================
# Print timing
# ============================================================

def print_timing(
    retrieval_timing,
    llm_time,
    prompt_tokens,
    generated_tokens
):

    total_time = (
        retrieval_timing["total_retrieval"]
        +
        llm_time
    )

    print()
    print("=" * 70)
    print("TIMING")
    print("=" * 70)

    print(
        f"Embedding/search: "
        f"{retrieval_timing['embedding']:.3f} s"
    )

    print(
        f"Reranking:        "
        f"{retrieval_timing['reranking']:.3f} s"
    )

    print(
        f"Total retrieval:  "
        f"{retrieval_timing['total_retrieval']:.3f} s"
    )

    print(
        f"LLM generation:   "
        f"{llm_time:.3f} s"
    )

    print(
        f"TOTAL:            "
        f"{total_time:.3f} s"
    )

    if prompt_tokens is not None:

        print(
            f"Prompt tokens:    "
            f"{prompt_tokens}"
        )

    if generated_tokens is not None:

        print(
            f"Generated tokens: "
            f"{generated_tokens}"
        )

    print()


# ============================================================
# Interactive application
# ============================================================

print()
print("=" * 70)
print("ROMANIAN RAG ASSISTANT")
print("=" * 70)
print(f"LLM: {LLM_NAME}")
print(
    f"Chunks sent to LLM: {FINAL_RESULTS}"
)
print(
    f"Maximum generated tokens: "
    f"{MAX_GENERATED_TOKENS}"
)
print("Type 'exit' to stop.")
print()


while True:

    try:

        query = input(
            "Întrebare: "
        ).strip()

    except (KeyboardInterrupt, EOFError):

        print("\nExiting...")
        break


    if query.lower() == "exit":

        print("Exiting...")
        break


    if not query:
        continue


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    (
        retrieved_chunks,
        category,
        retrieval_timing
    ) = retrieve(query)


    print()

    if category:

        print(
            f"Topic detectat: {category}"
        )

    else:

        print(
            "Topic detectat: general"
        )


    print()
    print("Fragmente recuperate:")


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"{i}. "
            f"{chunk['source']}, "
            f"pagina {chunk['page']} "
            f"(scor "
            f"{chunk['combined_score']:.4f})"
        )


    print()
    print("Se generează răspunsul...")


    try:

        (
            answer,
            llm_time,
            prompt_tokens,
            generated_tokens
        ) = generate_answer(
            query,
            retrieved_chunks
        )


        print()
        print("=" * 70)
        print("RĂSPUNS")
        print("=" * 70)

        print(answer)


        print_timing(
            retrieval_timing,
            llm_time,
            prompt_tokens,
            generated_tokens
        )


    except Exception as error:

        print()
        print("=" * 70)
        print("EROARE LLM")
        print("=" * 70)

        print(error)


    print()
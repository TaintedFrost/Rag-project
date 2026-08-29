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

SEARCH_RESULTS = 10
FINAL_RESULTS = 4

MAX_GENERATED_TOKENS = 220


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
# Topic keywords
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
        "înscriere la doctorat",
    ],

    "licenta": [
        "licență",
        "licenta",
        "studii de licență",
        "ciclu de licență",
        "admitere licență",
        "admiterea la licență",
        "concursul de admitere la licență",
        "bacalaureat",
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
# Useful phrases for evidence selection
# ============================================================

QUESTION_TYPE_KEYWORDS = {
    "conditions": [
        "condiții",
        "conditii",
        "criterii",
        "eligibilitate",
        "cerințe",
        "cerinte",
        "necesare",
    ],

    "documents": [
        "documente",
        "dosar",
        "acte",
        "ce trebuie depus",
        "ce document",
    ],

    "dates": [
        "când",
        "cand",
        "data",
        "perioada",
        "începe",
        "incepe",
        "se termină",
        "se termina",
    ],

    "numbers": [
        "cât",
        "cat",
        "număr",
        "numar",
        "punctaj",
        "medie",
        "credite",
        "durata",
    ],
}


# ============================================================
# Load metadata
# ============================================================

print("Loading RAG data...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} total chunks.")


# ============================================================
# Load embedding model
# ============================================================

print(
    "Loading embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

model_start = time.perf_counter()

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

model_load_time = (
    time.perf_counter() - model_start
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
        INDEX_FOLDER / f"{category}.index"
    )

    mapping_path = (
        INDEX_FOLDER / f"{category}_mapping.json"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing index: {index_path}\n"
            "Run create_category_indexes.py first."
        )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing mapping: {mapping_path}\n"
            "Run create_category_indexes.py first."
        )

    category_indexes[category] = (
        faiss.read_index(str(index_path))
    )

    with mapping_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        category_mappings[category] = json.load(file)


# ============================================================
# Text helpers
# ============================================================

def normalize_text(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def keyword_score(query, text):
    query_words = set(normalize_text(query))
    text_words = set(normalize_text(text))

    if not query_words:
        return 0.0

    return len(
        query_words.intersection(text_words)
    ) / len(query_words)


def split_sentences(text):
    parts = re.split(
        r"(?<=[.!?;])\s+|\n+",
        text
    )

    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= 25
    ]


# ============================================================
# Detect topic
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
# Detect question type
# ============================================================

def detect_question_types(query):

    query_lower = query.lower()

    detected = []

    for question_type, keywords in (
        QUESTION_TYPE_KEYWORDS.items()
    ):
        for keyword in keywords:
            if keyword in query_lower:
                detected.append(question_type)
                break

    return detected


# ============================================================
# Semantic search
# ============================================================

def semantic_search(
    query,
    category
):

    index = category_indexes[category]
    mapping = category_mappings[category]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
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
            "chunk_index": original_index,
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "semantic_score": float(score),
        })

    return results


# ============================================================
# Retrieve chunks
# ============================================================

def retrieve(query):

    retrieval_start = time.perf_counter()

    category = detect_category(query)

    # --------------------------------------------------------
    # Unknown-topic fallback
    # --------------------------------------------------------

    if category is None:

        global_index = faiss.read_index(
            "faiss_index.bin"
        )

        embedding_start = time.perf_counter()

        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )

        search_start = time.perf_counter()

        search_k = min(
            SEARCH_RESULTS,
            global_index.ntotal
        )

        scores, indices = global_index.search(
            query_embedding,
            search_k
        )

        search_time = (
            time.perf_counter()
            - search_start
        )

        candidates = []

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number < 0:
                continue

            chunk = chunks[
                index_number
            ]

            candidates.append({
                "chunk_index": index_number,
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "semantic_score": float(score),
                "keyword_score": keyword_score(
                    query,
                    chunk["text"]
                ),
            })

        candidates.sort(
            key=lambda x:
            (
                0.85 * x["semantic_score"]
                +
                0.15 * x["keyword_score"]
            ),
            reverse=True
        )

        selected = []

        seen_pages = set()

        for candidate in candidates:

            page_key = (
                candidate["source"],
                candidate["page"]
            )

            if page_key in seen_pages:
                continue

            seen_pages.add(page_key)

            candidate["combined_score"] = (
                0.85 * candidate["semantic_score"]
                +
                0.15 * candidate["keyword_score"]
            )

            selected.append(candidate)

            if len(selected) >= FINAL_RESULTS:
                break

        total_time = (
            time.perf_counter()
            - retrieval_start
        )

        return (
            selected,
            category,
            {
                "embedding": embedding_time,
                "search": search_time,
                "reranking": 0.0,
                "total_retrieval": total_time,
            }
        )


    # --------------------------------------------------------
    # Category-specific multi-query retrieval
    # --------------------------------------------------------

    search_queries = [query]

    if category in CATEGORY_KEYWORDS:

        if category == "burse":
            search_queries.append(
                "criterii bursa de performanță"
            )

        elif category == "doctorat":
            search_queries.append(
                "condiții înscriere doctorat"
            )

        elif category == "masterat":
            search_queries.append(
                "condiții admitere masterat"
            )

        elif category == "licenta":
            search_queries.append(
                "condiții admitere licență"
            )


    embedding_start = time.perf_counter()

    results_by_chunk = {}


    for search_query in search_queries:

        results = semantic_search(
            search_query,
            category
        )

        for result in results:

            index_number = (
                result["chunk_index"]
            )

            if index_number not in results_by_chunk:

                results_by_chunk[index_number] = {
                    "chunk_index": index_number,
                    "source": result["source"],
                    "page": result["page"],
                    "text": result["text"],
                    "semantic_scores": [],
                }

            results_by_chunk[
                index_number
            ]["semantic_scores"].append(
                result["semantic_score"]
            )


    embedding_time = (
        time.perf_counter()
        - embedding_start
    )


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    candidates = []

    question_types = detect_question_types(
        query
    )


    for candidate in (
        results_by_chunk.values()
    ):

        best_semantic = max(
            candidate["semantic_scores"]
        )

        lexical = keyword_score(
            query,
            candidate["text"]
        )

        combined = (
            0.80 * best_semantic
            +
            0.20 * lexical
        )

        candidates.append({
            "chunk_index":
                candidate["chunk_index"],

            "source":
                candidate["source"],

            "page":
                candidate["page"],

            "text":
                candidate["text"],

            "semantic_score":
                best_semantic,

            "keyword_score":
                lexical,

            "combined_score":
                combined,
        })


    candidates.sort(
        key=lambda x:
        x["combined_score"],
        reverse=True
    )


    reranking_time = (
        time.perf_counter()
        - rerank_start
    )


    # --------------------------------------------------------
    # Select unique pages
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

        seen_pages.add(page_key)

        selected.append(candidate)

        if len(selected) >= FINAL_RESULTS:
            break


    total_time = (
        time.perf_counter()
        - retrieval_start
    )


    return (
        selected,
        category,
        {
            "embedding": embedding_time,
            "search": 0.0,
            "reranking": reranking_time,
            "total_retrieval": total_time,
        }
    )


# ============================================================
# Evidence selection
# ============================================================

def select_evidence(
    query,
    retrieved_chunks
):

    evidence_start = time.perf_counter()

    question_words = set(
        normalize_text(query)
    )

    question_types = detect_question_types(
        query
    )

    evidence = []

    for chunk in retrieved_chunks:

        sentences = split_sentences(
            chunk["text"]
        )

        for sentence in sentences:

            sentence_words = set(
                normalize_text(sentence)
            )

            overlap = len(
                question_words.intersection(
                    sentence_words
                )
            )

            score = (
                0.65
                * chunk["semantic_score"]
                +
                0.35
                * (
                    overlap
                    /
                    max(len(question_words), 1)
                )
            )

            # Give a small boost to sentences containing
            # explicit requirement/number language for
            # questions asking for conditions or values.
            sentence_lower = (
                sentence.lower()
            )

            if (
                "conditions" in question_types
                or
                "numbers" in question_types
            ):

                if any(
                    marker in sentence_lower
                    for marker in [
                        "minimum",
                        "minim",
                        "cel puțin",
                        "cel putin",
                        "maximum",
                        "maxim",
                        "9,70",
                        "30 de puncte",
                        "300",
                        "credite",
                        "ani",
                    ]
                ):
                    score += 0.10


            evidence.append({
                "text": sentence,
                "source": chunk["source"],
                "page": chunk["page"],
                "score": score,
            })


    evidence.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    selected = []

    seen_text = set()


    for item in evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower()
        )

        if normalized in seen_text:
            continue

        seen_text.add(normalized)

        selected.append(item)

        if len(selected) >= 8:
            break


    evidence_time = (
        time.perf_counter()
        - evidence_start
    )

    return selected, evidence_time


# ============================================================
# Build LLM prompt
# ============================================================

def build_prompt(
    query,
    evidence
):

    context_parts = []

    for i, item in enumerate(
        evidence,
        start=1
    ):

        context_parts.append(
            f"""
DOVADĂ {i}
Sursă: {item['source']}
Pagina: {item['page']}

{item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    return f"""
/no_think

Răspunde la întrebarea de mai jos folosind
EXCLUSIV dovezile furnizate.

Întrebare:
{query}

Reguli:
- Răspunde numai în limba română.
- Răspunde direct și clar.
- Include toate informațiile relevante din dovezi.
- Pentru întrebări despre condiții sau cifre,
  include valorile exacte.
- Nu inventa informații.
- Nu adăuga informații generale.
- Nu inventa surse.
- Nu genera o secțiune de surse.
- Dacă informația nu apare în dovezi, spune:
"Nu am găsit această informație în documentele disponibile."

Dovezi:
{context}

Răspuns:
"""


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    query,
    evidence
):

    start = time.perf_counter()

    prompt = build_prompt(
        query,
        evidence
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

    elapsed = (
        time.perf_counter()
        - start
    )

    answer = (
        response.message.content
        .strip()
    )

    return (
        answer,
        elapsed,
        getattr(
            response,
            "prompt_eval_count",
            None
        ),
        getattr(
            response,
            "eval_count",
            None
        ),
    )


# ============================================================
# Build source list
# ============================================================

def build_sources(
    retrieved_chunks
):

    sources = []

    seen = set()

    for chunk in retrieved_chunks:

        key = (
            chunk["source"],
            chunk["page"]
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            f"- {chunk['source']}, "
            f"pagina {chunk['page']}"
        )

    return sources


# ============================================================
# Interactive application
# ============================================================

print()
print("=" * 70)
print("ROMANIAN RAG ASSISTANT")
print("=" * 70)
print(f"LLM: {LLM_NAME}")
print(
    f"Retrieved chunks: {FINAL_RESULTS}"
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


    # --------------------------------------------------------
    # Evidence selection
    # --------------------------------------------------------

    evidence, evidence_time = (
        select_evidence(
            query,
            retrieved_chunks
        )
    )


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
    print("Dovezi selectate:")


    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"{i}. "
            f"{item['source']}, "
            f"pagina {item['page']} "
            f"(scor "
            f"{item['score']:.4f})"
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
            evidence
        )


        sources = build_sources(
            retrieved_chunks
        )


        print()
        print("=" * 70)
        print("RĂSPUNS")
        print("=" * 70)

        print(answer)


        print()
        print("Surse:")

        for source in sources:

            print(source)


        total_time = (
            retrieval_timing[
                "total_retrieval"
            ]
            +
            evidence_time
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
            f"Evidence select:  "
            f"{evidence_time:.3f} s"
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


    except Exception as error:

        print()
        print("=" * 70)
        print("EROARE")
        print("=" * 70)

        print(error)


    print()
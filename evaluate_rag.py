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

QUESTIONS_PATH = "evaluation_questions.json"
METADATA_PATH = "chunk_metadata.json"

INDEX_FOLDER = Path("indexes")

SEARCH_RESULTS = 10
FINAL_RESULTS = 3

MAX_GENERATED_TOKENS = 180

RESULTS_PATH = "evaluation_results.json"


# ============================================================
# Category configuration
# ============================================================

CATEGORY_DOCUMENTS = {
    "doctorat": "doc1.pdf",
    "licenta": "doc2.pdf",
    "burse": "doc3.pdf",
    "masterat": "doc4.pdf",
}


# ============================================================
# Load evaluation questions
# ============================================================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:

    evaluation_questions = json.load(file)


print(
    f"Loaded {len(evaluation_questions)} evaluation questions."
)


# ============================================================
# Load chunks
# ============================================================

print("Loading chunk metadata...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


print(
    f"Loaded {len(chunks)} chunks."
)


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
    time.perf_counter()
    - model_start
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


for category, source in CATEGORY_DOCUMENTS.items():

    index_path = (
        INDEX_FOLDER /
        f"{category}.index"
    )

    mapping_path = (
        INDEX_FOLDER /
        f"{category}_mapping.json"
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
        query_words
        &
        text_words
    )

    return (
        len(overlap)
        /
        len(query_words)
    )


# ============================================================
# Retrieve chunks
# ============================================================
def retrieve(
    question,
    category
):

    retrieval_start = time.perf_counter()


    # --------------------------------------------------------
    # General / unanswerable questions
    #
    # Search the global index rather than requiring a
    # category-specific index.
    # --------------------------------------------------------

    if category == "general":

        global_index = faiss.read_index(
            "faiss_index.bin"
        )

        query_embedding_start = (
            time.perf_counter()
        )

        query_embedding = (
            embedding_model.encode(
                [question],
                normalize_embeddings=True
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        embedding_time = (
            time.perf_counter()
            -
            query_embedding_start
        )


        search_start = (
            time.perf_counter()
        )

        search_k = min(
            SEARCH_RESULTS,
            global_index.ntotal
        )

        semantic_scores, indices = (
            global_index.search(
                query_embedding,
                search_k
            )
        )

        search_time = (
            time.perf_counter()
            -
            search_start
        )


        candidates = []


        for score, index_number in zip(
            semantic_scores[0],
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
                    question,
                    chunk["text"]
                ),
                "combined_score": float(score)
            })


        candidates.sort(
            key=lambda item:
            item["combined_score"],
            reverse=True
        )


        # Keep unique source/page combinations.
        final_candidates = []

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

            final_candidates.append(
                candidate
            )

            if len(final_candidates) >= FINAL_RESULTS:
                break


        total_time = (
            time.perf_counter()
            -
            retrieval_start
        )


        return (
            final_candidates,
            {
                "embedding": embedding_time,
                "faiss_search": search_time,
                "reranking": 0.0,
                "total_retrieval": total_time
            }
        )


    # --------------------------------------------------------
    # Category-specific questions
    # --------------------------------------------------------

    if category not in category_indexes:

        raise ValueError(
            f"Unknown category: {category}"
        )


    index = category_indexes[
        category
    ]

    mapping = category_mappings[
        category
    ]


    # --------------------------------------------------------
    # Question embedding
    # --------------------------------------------------------

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    embedding_time = (
        time.perf_counter()
        -
        embedding_start
    )


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    search_start = (
        time.perf_counter()
    )

    search_k = min(
        SEARCH_RESULTS,
        index.ntotal
    )

    semantic_scores, indices = (
        index.search(
            query_embedding,
            search_k
        )
    )

    search_time = (
        time.perf_counter()
        -
        search_start
    )


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    rerank_start = (
        time.perf_counter()
    )

    candidates = []


    for semantic_score, local_index in zip(
        semantic_scores[0],
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


        lexical = keyword_score(
            question,
            chunk["text"]
        )


        combined = (
            0.85
            * float(semantic_score)
            +
            0.15
            * lexical
        )


        candidates.append({
            "chunk_index": original_index,
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "semantic_score": float(
                semantic_score
            ),
            "keyword_score": lexical,
            "combined_score": combined
        })


    candidates.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Remove duplicate pages
    # --------------------------------------------------------

    final_candidates = []

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

        final_candidates.append(
            candidate
        )

        if len(final_candidates) >= FINAL_RESULTS:
            break


    rerank_time = (
        time.perf_counter()
        -
        rerank_start
    )


    total_time = (
        time.perf_counter()
        -
        retrieval_start
    )


    return (
        final_candidates,
        {
            "embedding": embedding_time,
            "faiss_search": search_time,
            "reranking": rerank_time,
            "total_retrieval": total_time
        }
    )


# ============================================================
# Build LLM prompt
# ============================================================

def build_prompt(
    question,
    retrieved_chunks
):

    context = []


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context.append(
            f"""
CONTEXT {i}

Sursa: {chunk['source']}
Pagina: {chunk['page']}

{chunk['text']}
"""
        )


    context_text = "\n".join(
        context
    )


    return f"""
/no_think

Răspunde la întrebarea de mai jos folosind
EXCLUSIV informațiile din context.

Întrebare:
{question}

Reguli:
- Răspunde în limba română.
- Răspunde direct și concis.
- Include toate condițiile relevante.
- Păstrează valorile numerice exacte.
- Nu inventa informații.
- Dacă răspunsul nu poate fi găsit în context,
  spune exact:
"Nu am găsit această informație în documentele disponibile."

La final scrie sursele folosite:

Surse:
Folosește exact numele fișierului și pagina din context.

CONTEXT:
{context_text}

Răspuns:
"""


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    prompt = build_prompt(
        question,
        retrieved_chunks
    )


    start = time.perf_counter()


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
            "num_predict": MAX_GENERATED_TOKENS
        }
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    answer = response.message.content


    generated_tokens = getattr(
        response,
        "eval_count",
        None
    )


    prompt_tokens = getattr(
        response,
        "prompt_eval_count",
        None
    )


    return (
        answer,
        elapsed,
        prompt_tokens,
        generated_tokens
    )


# ============================================================
# Check retrieval
# ============================================================

def evaluate_retrieval(
    retrieved_chunks,
    expected_source,
    expected_pages
):

    sources = [
        chunk["source"]
        for chunk in retrieved_chunks
    ]


    pages = [
        chunk["page"]
        for chunk in retrieved_chunks
    ]


    if expected_source is None:

        # For an unanswerable question, there isn't
        # supposed to be a correct source.
        return {
            "source_correct": True,
            "page_correct": True
        }


    source_correct = (
        expected_source
        in sources
    )


    page_correct = any(
        page in pages
        for page in expected_pages
    )


    return {
        "source_correct": source_correct,
        "page_correct": page_correct
    }


# ============================================================
# Check answer
# ============================================================

def evaluate_answer(
    answer,
    expected_keywords,
    should_refuse
):

    answer_lower = answer.lower()


    # --------------------------------------------------------
    # Refusal test
    # --------------------------------------------------------

    refusal_phrases = [
        "nu am găsit această informație",
        "nu am găsit informația",
        "informația nu apare",
        "nu este disponibilă",
        "nu poate fi determinat"
    ]


    refused = any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


    if should_refuse:

        return {
            "answer_correct": refused,
            "keyword_coverage": 1.0
            if refused else 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "refused": refused
        }


    # --------------------------------------------------------
    # Normal answer test
    # --------------------------------------------------------

    matched = []
    missing = []


    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            matched.append(keyword)

        else:

            missing.append(keyword)


    if expected_keywords:

        coverage = (
            len(matched)
            /
            len(expected_keywords)
        )

    else:

        coverage = 1.0


    # We consider the generated answer acceptable
    # when at least 60% of the expected keywords
    # are present.
    answer_correct = (
        coverage >= 0.60
        and not refused
    )


    return {
        "answer_correct": answer_correct,
        "keyword_coverage": coverage,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "refused": refused
    }


# ============================================================
# Run evaluation
# ============================================================

results = []


print()
print("=" * 80)
print("RAG AUTOMATIC EVALUATION")
print("=" * 80)


evaluation_start = time.perf_counter()


for question_data in evaluation_questions:

    question_id = question_data["id"]

    question = question_data["question"]

    category = question_data["category"]

    expected_source = (
        question_data["expected_source"]
    )

    expected_pages = (
        question_data["expected_pages"]
    )

    expected_keywords = (
        question_data["expected_keywords"]
    )

    should_refuse = (
        question_data["should_refuse"]
    )


    print()
    print("=" * 80)
    print(
        f"{question_id}: {question}"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_chunks, retrieval_timing = (
        retrieve(
            question,
            category
        )
    )


    print()
    print("Retrieved:")


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"{i}. "
            f"{chunk['source']}, "
            f"pagina {chunk['page']} "
            f"(score "
            f"{chunk['combined_score']:.4f})"
        )


    retrieval_evaluation = (
        evaluate_retrieval(
            retrieved_chunks,
            expected_source,
            expected_pages
        )
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print()
    print("Generating answer...")


    (
        answer,
        llm_time,
        prompt_tokens,
        generated_tokens
    ) = generate_answer(
        question,
        retrieved_chunks
    )


    print()
    print("Answer:")
    print(answer)


    # --------------------------------------------------------
    # Answer evaluation
    # --------------------------------------------------------

    answer_evaluation = (
        evaluate_answer(
            answer,
            expected_keywords,
            should_refuse
        )
    )


    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    retrieval_score = (
        1.0
        if (
            retrieval_evaluation[
                "source_correct"
            ]
            and
            retrieval_evaluation[
                "page_correct"
            ]
        )
        else 0.0
    )


    answer_score = (
        answer_evaluation[
            "keyword_coverage"
        ]
    )


    total_score = (
        0.50 * retrieval_score
        +
        0.50 * answer_score
    )


    result = {
        "id": question_id,
        "question": question,
        "category": category,

        "expected_source": expected_source,
        "expected_pages": expected_pages,

        "retrieved_sources": [
            chunk["source"]
            for chunk in retrieved_chunks
        ],

        "retrieved_pages": [
            chunk["page"]
            for chunk in retrieved_chunks
        ],

        "source_correct": retrieval_evaluation[
            "source_correct"
        ],

        "page_correct": retrieval_evaluation[
            "page_correct"
        ],

        "retrieval_correct": (
            retrieval_score == 1.0
        ),

        "answer": answer,

        "expected_keywords": expected_keywords,

        "matched_keywords": answer_evaluation[
            "matched_keywords"
        ],

        "missing_keywords": answer_evaluation[
            "missing_keywords"
        ],

        "keyword_coverage": answer_evaluation[
            "keyword_coverage"
        ],

        "answer_correct": answer_evaluation[
            "answer_correct"
        ],

        "should_refuse": should_refuse,

        "refused": answer_evaluation[
            "refused"
        ],

        "retrieval_time_seconds": (
            retrieval_timing[
                "total_retrieval"
            ]
        ),

        "llm_time_seconds": llm_time,

        "prompt_tokens": prompt_tokens,

        "generated_tokens": generated_tokens,

        "total_score": total_score
    }


    results.append(result)


# ============================================================
# Overall metrics
# ============================================================

evaluation_time = (
    time.perf_counter()
    -
    evaluation_start
)


total_questions = len(results)


retrieval_correct = sum(
    1
    for result in results
    if result["retrieval_correct"]
)


answer_correct = sum(
    1
    for result in results
    if result["answer_correct"]
)


avg_keyword_coverage = (
    sum(
        result["keyword_coverage"]
        for result in results
    )
    /
    total_questions
)


avg_retrieval_time = (
    sum(
        result["retrieval_time_seconds"]
        for result in results
    )
    /
    total_questions
)


avg_llm_time = (
    sum(
        result["llm_time_seconds"]
        for result in results
    )
    /
    total_questions
)


avg_total_score = (
    sum(
        result["total_score"]
        for result in results
    )
    /
    total_questions
)


# ============================================================
# Print summary
# ============================================================

print()
print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print(
    f"Questions:                 "
    f"{total_questions}"
)

print(
    f"Retrieval accuracy:        "
    f"{retrieval_correct / total_questions:.2%}"
)

print(
    f"Answer accuracy:           "
    f"{answer_correct / total_questions:.2%}"
)

print(
    f"Average keyword coverage:  "
    f"{avg_keyword_coverage:.2%}"
)

print(
    f"Average retrieval time:    "
    f"{avg_retrieval_time:.3f} s"
)

print(
    f"Average LLM time:          "
    f"{avg_llm_time:.3f} s"
)

print(
    f"Average total score:       "
    f"{avg_total_score:.2%}"
)

print(
    f"Total evaluation time:     "
    f"{evaluation_time:.2f} s"
)


# ============================================================
# Per-question summary
# ============================================================

print()
print("=" * 80)
print("PER-QUESTION RESULTS")
print("=" * 80)


for result in results:

    retrieval_status = (
        "PASS"
        if result["retrieval_correct"]
        else "FAIL"
    )

    answer_status = (
        "PASS"
        if result["answer_correct"]
        else "FAIL"
    )


    print(
        f"{result['id']} | "
        f"Retrieval: {retrieval_status} | "
        f"Answer: {answer_status} | "
        f"Score: {result['total_score']:.2%}"
    )


# ============================================================
# Save results
# ============================================================

report = {
    "configuration": {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm": LLM_NAME,
        "questions": total_questions,
        "search_results": SEARCH_RESULTS,
        "final_results": FINAL_RESULTS,
        "max_generated_tokens": MAX_GENERATED_TOKENS
    },

    "summary": {
        "retrieval_accuracy": (
            retrieval_correct
            /
            total_questions
        ),

        "answer_accuracy": (
            answer_correct
            /
            total_questions
        ),

        "average_keyword_coverage":
            avg_keyword_coverage,

        "average_retrieval_time_seconds":
            avg_retrieval_time,

        "average_llm_time_seconds":
            avg_llm_time,

        "average_total_score":
            avg_total_score,

        "total_evaluation_time_seconds":
            evaluation_time
    },

    "questions": results
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=2
    )


print()
print(
    f"Detailed results saved to: "
    f"{RESULTS_PATH}"
)
print()
print("Evaluation complete.")
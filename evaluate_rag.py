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
FINAL_RESULTS = 4
MAX_GENERATED_TOKENS = 220

RESULTS_PATH = "evaluation_results.json"


# ============================================================
# Categories
# ============================================================

CATEGORY_DOCUMENTS = {
    "doctorat": "doc1.pdf",
    "licenta": "doc2.pdf",
    "burse": "doc3.pdf",
    "masterat": "doc4.pdf",
}


# ============================================================
# Load files
# ============================================================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    evaluation_questions = json.load(file)


print(
    f"Loaded {len(evaluation_questions)} "
    f"evaluation questions."
)


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
    -
    model_start
)

print(
    f"Embedding model loaded in "
    f"{model_load_time:.2f} seconds."
)


# ============================================================
# Load FAISS indexes
# ============================================================

category_indexes = {}
category_mappings = {}


for category in CATEGORY_DOCUMENTS:

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
            f"Missing index: {index_path}"
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


def keyword_score(
    query,
    text
):

    query_words = set(
        normalize_text(query)
    )

    text_words = set(
        normalize_text(text)
    )

    if not query_words:

        return 0.0


    return (
        len(
            query_words
            &
            text_words
        )
        /
        len(query_words)
    )


def split_sentences(text):

    pieces = re.split(
        r"(?<=[.!?;])\s+|\n+",
        text
    )

    return [
        piece.strip()
        for piece in pieces
        if len(piece.strip()) >= 25
    ]


# ============================================================
# Retrieve
# ============================================================

def retrieve(
    question,
    category
):

    start = time.perf_counter()


    # --------------------------------------------------------
    # Unknown question
    # --------------------------------------------------------

    if category == "general":

        global_index = faiss.read_index(
            "faiss_index.bin"
        )


        embedding_start = time.perf_counter()


        embedding = (
            embedding_model.encode(
                [question],
                normalize_embeddings=True
            )
        )


        embedding = np.asarray(
            embedding,
            dtype="float32"
        )


        embedding_time = (
            time.perf_counter()
            -
            embedding_start
        )


        search_start = time.perf_counter()


        search_k = min(
            SEARCH_RESULTS,
            global_index.ntotal
        )


        scores, indices = (
            global_index.search(
                embedding,
                search_k
            )
        )


        search_time = (
            time.perf_counter()
            -
            search_start
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
                "chunk_index":
                    index_number,

                "source":
                    chunk["source"],

                "page":
                    chunk["page"],

                "text":
                    chunk["text"],

                "semantic_score":
                    float(score),

                "keyword_score":
                    keyword_score(
                        question,
                        chunk["text"]
                    ),

                "combined_score":
                    float(score)
            })


        results.sort(
            key=lambda item:
            item["combined_score"],
            reverse=True
        )


        total_time = (
            time.perf_counter()
            - start
        )


        return (
            results[:FINAL_RESULTS],
            {
                "embedding":
                    embedding_time,

                "faiss_search":
                    search_time,

                "reranking":
                    0.0,

                "total_retrieval":
                    total_time
            }
        )


    # --------------------------------------------------------
    # Category retrieval
    # --------------------------------------------------------

    index = category_indexes[
        category
    ]

    mapping = category_mappings[
        category
    ]


    embedding_start = time.perf_counter()


    embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
    )


    embedding = np.asarray(
        embedding,
        dtype="float32"
    )


    embedding_time = (
        time.perf_counter()
        -
        embedding_start
    )


    search_start = time.perf_counter()


    search_k = min(
        SEARCH_RESULTS,
        index.ntotal
    )


    scores, indices = (
        index.search(
            embedding,
            search_k
        )
    )


    search_time = (
        time.perf_counter()
        -
        search_start
    )


    candidates = []


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


        lexical = keyword_score(
            question,
            chunk["text"]
        )


        combined = (
            0.85 * float(score)
            +
            0.15 * lexical
        )


        candidates.append({

            "chunk_index":
                original_index,

            "source":
                chunk["source"],

            "page":
                chunk["page"],

            "text":
                chunk["text"],

            "semantic_score":
                float(score),

            "keyword_score":
                lexical,

            "combined_score":
                combined
        })


    candidates.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Unique pages
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
        search_start
        -
        search_time
    )


    total_time = (
        time.perf_counter()
        -
        start
    )


    return (
        selected,
        {
            "embedding":
                embedding_time,

            "faiss_search":
                search_time,

            "reranking":
                reranking_time,

            "total_retrieval":
                total_time
        }
    )


# ============================================================
# Evidence selection
# ============================================================

def select_evidence(
    question,
    retrieved_chunks
):

    evidence = []


    question_words = set(
        normalize_text(question)
    )


    for chunk in retrieved_chunks:

        for sentence in split_sentences(
            chunk["text"]
        ):

            sentence_words = set(
                normalize_text(sentence)
            )


            overlap = (
                len(
                    question_words
                    &
                    sentence_words
                )
                /
                max(
                    len(question_words),
                    1
                )
            )


            score = (
                0.65
                * chunk["semantic_score"]
                +
                0.35
                * overlap
            )


            evidence.append({

                "text":
                    sentence,

                "source":
                    chunk["source"],

                "page":
                    chunk["page"],

                "score":
                    score
            })


    evidence.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    selected = []

    seen = set()


    for item in evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower()
        )


        if normalized in seen:

            continue


        seen.add(
            normalized
        )


        selected.append(
            item
        )


        if len(selected) >= 8:

            break


    return selected


# ============================================================
# Build prompt
# ============================================================

def build_prompt(
    question,
    evidence
):

    context = []


    for i, item in enumerate(
        evidence,
        start=1
    ):

        context.append(
            f"""
DOVADĂ {i}
Sursă: {item['source']}
Pagina: {item['page']}

{item['text']}
"""
        )


    return f"""
/no_think

Răspunde la întrebarea de mai jos
folosind EXCLUSIV informațiile din dovezi.

Întrebare:
{question}

Reguli:
- Răspunde în limba română.
- Răspunde direct.
- Include toate condițiile relevante.
- Păstrează valorile numerice exacte.
- Nu inventa informații.
- Nu adăuga informații generale.
- Nu genera surse.
- Dacă informația nu apare în dovezi, spune:
"Nu am găsit această informație în documentele disponibile."

DOVEZI:
{"".join(context)}

Răspuns:
"""


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    question,
    evidence
):

    prompt = build_prompt(
        question,
        evidence
    )


    start = time.perf_counter()


    response = chat(
        model=LLM_NAME,
        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],
        think=False,
        options={
            "temperature": 0.1,
            "num_predict":
                MAX_GENERATED_TOKENS
        }
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    return (
        response.message.content.strip(),
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
        )
    )


# ============================================================
# Evaluate retrieval
# ============================================================

def evaluate_retrieval(
    retrieved,
    expected_source,
    expected_pages
):

    sources = [
        item["source"]
        for item in retrieved
    ]


    pages = [
        item["page"]
        for item in retrieved
    ]


    if expected_source is None:

        return {
            "source_correct": True,
            "page_recall": True
        }


    return {

        "source_correct":
            expected_source in sources,

        "page_recall":
            any(
                page in pages
                for page in expected_pages
            )
    }


# ============================================================
# Evaluate answer
# ============================================================

def evaluate_answer(
    answer,
    expected_keywords,
    should_refuse
):

    answer_lower = answer.lower()


    refusal_phrases = [
        "nu am găsit această informație",
        "nu am găsit informația",
        "informația nu apare",
        "nu este disponibilă"
    ]


    refused = any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


    if should_refuse:

        return {
            "answer_correct":
                refused,

            "keyword_coverage":
                1.0 if refused else 0.0,

            "matched_keywords": [],

            "missing_keywords": [],

            "refused":
                refused
        }


    matched = []

    missing = []


    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            matched.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )


    coverage = (
        len(matched)
        /
        len(expected_keywords)
        if expected_keywords
        else 1.0
    )


    return {

        "answer_correct":
            len(missing) == 0
            and not refused,

        "keyword_coverage":
            coverage,

        "matched_keywords":
            matched,

        "missing_keywords":
            missing,

        "refused":
            refused
    }


# ============================================================
# Evaluation
# ============================================================

results = []

evaluation_start = time.perf_counter()


print()
print("=" * 80)
print("RAG AUTOMATIC EVALUATION")
print("=" * 80)


for item in evaluation_questions:

    question_id = item["id"]
    question = item["question"]
    category = item["category"]

    expected_source = (
        item["expected_source"]
    )

    expected_pages = (
        item["expected_pages"]
    )

    expected_keywords = (
        item["expected_keywords"]
    )

    should_refuse = (
        item["should_refuse"]
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

    (
        retrieved,
        retrieval_timing
    ) = retrieve(
        question,
        category
    )


    print()
    print("Retrieved:")


    for i, result in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['source']}, "
            f"pagina {result['page']} "
            f"(score "
            f"{result['combined_score']:.4f})"
        )


    retrieval_eval = (
        evaluate_retrieval(
            retrieved,
            expected_source,
            expected_pages
        )
    )


    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence = select_evidence(
        question,
        retrieved
    )


    print()
    print("Evidence passages:")


    for i, evidence_item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"{i}. "
            f"{evidence_item['source']}, "
            f"pagina "
            f"{evidence_item['page']}"
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
        evidence
    )


    print()
    print("Answer:")
    print(answer)


    # --------------------------------------------------------
    # Answer evaluation
    # --------------------------------------------------------

    answer_eval = evaluate_answer(
        answer,
        expected_keywords,
        should_refuse
    )


    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    retrieval_score = (
        1.0
        if (
            retrieval_eval[
                "source_correct"
            ]
            and
            retrieval_eval[
                "page_recall"
            ]
        )
        else 0.0
    )


    total_score = (
        0.50 * retrieval_score
        +
        0.50
        * answer_eval[
            "keyword_coverage"
        ]
    )


    results.append({

        "id":
            question_id,

        "question":
            question,

        "category":
            category,

        "expected_source":
            expected_source,

        "expected_pages":
            expected_pages,

        "retrieved_sources": [
            item["source"]
            for item in retrieved
        ],

        "retrieved_pages": [
            item["page"]
            for item in retrieved
        ],

        "source_correct":
            retrieval_eval[
                "source_correct"
            ],

        "page_recall":
            retrieval_eval[
                "page_recall"
            ],

        "answer":
            answer,

        "expected_keywords":
            expected_keywords,

        "matched_keywords":
            answer_eval[
                "matched_keywords"
            ],

        "missing_keywords":
            answer_eval[
                "missing_keywords"
            ],

        "keyword_coverage":
            answer_eval[
                "keyword_coverage"
            ],

        "answer_correct":
            answer_eval[
                "answer_correct"
            ],

        "should_refuse":
            should_refuse,

        "refused":
            answer_eval[
                "refused"
            ],

        "retrieval_time_seconds":
            retrieval_timing[
                "total_retrieval"
            ],

        "llm_time_seconds":
            llm_time,

        "prompt_tokens":
            prompt_tokens,

        "generated_tokens":
            generated_tokens,

        "total_score":
            total_score
    })


# ============================================================
# Summary
# ============================================================

evaluation_time = (
    time.perf_counter()
    -
    evaluation_start
)


total_questions = len(results)


source_accuracy = (
    sum(
        result["source_correct"]
        for result in results
    )
    /
    total_questions
)


page_recall = (
    sum(
        result["page_recall"]
        for result in results
    )
    /
    total_questions
)


answer_accuracy = (
    sum(
        result["answer_correct"]
        for result in results
    )
    /
    total_questions
)


refusal_accuracy = (
    sum(
        result["refused"] == result["should_refuse"]
        for result in results
    )
    /
    total_questions
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
    f"Document retrieval:        "
    f"{source_accuracy:.2%}"
)


print(
    f"Evidence/page recall:      "
    f"{page_recall:.2%}"
)


print(
    f"Answer accuracy:           "
    f"{answer_accuracy:.2%}"
)


print(
    f"Refusal accuracy:          "
    f"{refusal_accuracy:.2%}"
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
# Per-question results
# ============================================================

print()
print("=" * 80)
print("PER-QUESTION RESULTS")
print("=" * 80)


for result in results:

    retrieval_status = (
        "PASS"
        if (
            result["source_correct"]
            and
            result["page_recall"]
        )
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
        f"Coverage: "
        f"{result['keyword_coverage']:.0%}"
    )


# ============================================================
# Save results
# ============================================================

report = {

    "configuration": {

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "llm":
            LLM_NAME,

        "questions":
            total_questions,

        "search_results":
            SEARCH_RESULTS,

        "final_results":
            FINAL_RESULTS,

        "max_generated_tokens":
            MAX_GENERATED_TOKENS
    },


    "summary": {

        "document_retrieval_accuracy":
            source_accuracy,

        "evidence_page_recall":
            page_recall,

        "answer_accuracy":
            answer_accuracy,

        "refusal_accuracy":
            refusal_accuracy,

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


    "questions":
        results
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
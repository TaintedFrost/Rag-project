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

SEARCH_RESULTS = 50
FINAL_RESULTS = 4

MAX_GENERATED_TOKENS = 220

RESULTS_PATH = "evaluation_results.json"


# ============================================================
# Document categories
# ============================================================

CATEGORY_DOCUMENTS = {
    "doctorat": "doc1.pdf",
    "licenta": "doc2.pdf",
    "burse": "doc3.pdf",
    "masterat": "doc4.pdf",
}


# ============================================================
# Romanian stopwords
# ============================================================

STOPWORDS = {
    "care",
    "sunt",
    "este",
    "pentru",
    "la",
    "și",
    "si",
    "de",
    "din",
    "în",
    "in",
    "a",
    "al",
    "ale",
    "un",
    "o",
    "unui",
    "unei",
    "ce",
    "cum",
    "se",
    "pe",
    "cu",
    "prin",
    "sau",
    "fi",
    "mai",
    "ca",
}


# ============================================================
# Question type detection
# ============================================================

def detect_question_types(query):
    """
    Determine what kind of information the question asks for.
    This is used to make evidence selection more focused.
    """

    q = query.lower()

    types = []

    # --------------------------------------------------------
    # Number / quantitative questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "cât",
            "cat",
            "număr",
            "numar",
            "punctaj",
            "credite",
            "durată",
            "durata",
            "minim",
            "minimum",
            "maxim",
            "maximum",
            "câte",
            "cate",
        ]
    ):
        types.append("number")


    # --------------------------------------------------------
    # Date / period questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "când",
            "cand",
            "data",
            "perioada",
            "începe",
            "incepe",
            "început",
            "termină",
            "termina",
            "sfârșit",
            "sfarsit",
        ]
    ):
        types.append("date")


    # --------------------------------------------------------
    # Condition questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "condiții",
            "conditii",
            "criterii",
            "eligibilitate",
            "cerințe",
            "cerinte",
            "necesare",
            "necesar",
        ]
    ):
        types.append("conditions")


    # --------------------------------------------------------
    # Document questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "documente",
            "document",
            "dosar",
            "acte",
            "ce trebuie depus",
            "ce se depune",
        ]
    ):
        types.append("documents")


    return types


# ============================================================
# Text helpers
# ============================================================

def normalize_words(text):
    """
    Tokenize text and remove common Romanian stopwords.
    """

    words = re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


def lexical_score(query, text):
    """
    Basic lexical overlap between question and chunk.
    """

    query_words = set(
        normalize_words(query)
    )

    text_words = set(
        normalize_words(text)
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


def phrase_score(query, text):
    """
    Give a small boost when an important phrase from the
    question also appears directly in the retrieved text.
    """

    q = query.lower()
    t = text.lower()

    phrases = [
        "bursa de performanță",
        "bursa de performanta",
        "punctaj minim",
        "credite transferabile",
        "300 de credite",
        "înmatriculare",
        "inmatriculare",
        "validare",
        "bacalaureat",
        "criterii de departajare",
    ]

    score = 0.0

    for phrase in phrases:

        if (
            phrase in q
            and
            phrase in t
        ):
            score += 1.0

    return min(
        score,
        1.0
    )


# ============================================================
# Load evaluation questions
# ============================================================

print(
    f"Loading evaluation questions "
    f"from {QUESTIONS_PATH}..."
)

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:

    evaluation_questions = json.load(
        file
    )

print(
    f"Loaded {len(evaluation_questions)} "
    f"evaluation questions."
)


# ============================================================
# Load chunk metadata
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

print()
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
# Load category indexes
# ============================================================

print()
print("Loading category indexes...")

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


    if not mapping_path.exists():

        raise FileNotFoundError(
            f"Missing mapping: {mapping_path}"
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


print("Category indexes loaded.")


# ============================================================
# Retrieve chunks
# ============================================================

def retrieve(
    question,
    category
):
    """
    Same retrieval pipeline used by the interactive RAG:

        query
          ↓
        embedding
          ↓
        FAISS top-50
          ↓
        semantic + lexical + phrase reranking
          ↓
        top-4 unique pages
    """

    retrieval_start = time.perf_counter()


    # --------------------------------------------------------
    # Select the appropriate index
    # --------------------------------------------------------

    if category == "general":

        index = faiss.read_index(
            "faiss_index.bin"
        )

        mapping = list(
            range(len(chunks))
        )

    else:

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
    # Embed query
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


    for semantic, local_index in zip(
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


        lexical = lexical_score(
            question,
            chunk["text"]
        )


        phrase = phrase_score(
            question,
            chunk["text"]
        )


        combined = (
            0.70 * float(semantic)
            +
            0.20 * lexical
            +
            0.10 * phrase
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
                float(semantic),

            "keyword_score":
                lexical,

            "phrase_score":
                phrase,

            "combined_score":
                combined
        })


    candidates.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Keep one chunk per page
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
                total_retrieval
        }
    )


# ============================================================
# Targeted evidence extraction
# ============================================================

def extract_targeted_evidence(
    question,
    retrieved_chunks
):
    """
    Extract small text windows around terms relevant to the
    question type.

    This deliberately does NOT perform another LLM call.
    """

    question_types = (
        detect_question_types(question)
    )


    question_words = set(
        normalize_words(question)
    )


    search_terms = set()


    # --------------------------------------------------------
    # Numeric questions
    # --------------------------------------------------------

    if "number" in question_types:

        search_terms.update([
            "credite",
            "ects",
            "puncte",
            "punctaj",
            "minim",
            "minimum",
            "maxim",
            "maximum",
            "ani",
            "medie",
        ])


    # --------------------------------------------------------
    # Date questions
    # --------------------------------------------------------

    if "date" in question_types:

        search_terms.update([
            "începe",
            "incepe",
            "termină",
            "termina",
            "perioada",
            "data",
            "ianuarie",
            "februarie",
            "martie",
            "aprilie",
            "mai",
            "iunie",
            "iulie",
            "august",
            "septembrie",
            "octombrie",
            "noiembrie",
            "decembrie",
        ])


    # --------------------------------------------------------
    # Document questions
    # --------------------------------------------------------

    if "documents" in question_types:

        search_terms.update([
            "cerere",
            "curriculum vitae",
            "cv",
            "document",
            "dosar",
            "diplom",
            "supliment",
            "lista de lucrări",
            "lista",
        ])


    # --------------------------------------------------------
    # Condition questions
    # --------------------------------------------------------

    if "conditions" in question_types:

        search_terms.update([
            "condi",
            "criter",
            "trebuie",
            "necesar",
            "minimum",
            "minim",
            "eligibil",
            "se poate",
        ])


    # --------------------------------------------------------
    # Add meaningful words from the actual question
    # --------------------------------------------------------

    search_terms.update(
        word
        for word in question_words
        if len(word) >= 4
    )


    evidence = []


    # --------------------------------------------------------
    # Search every retrieved chunk
    # --------------------------------------------------------

    for chunk in retrieved_chunks:

        compact_text = re.sub(
            r"\s+",
            " ",
            chunk["text"]
        )


        lower_text = (
            compact_text.lower()
        )


        positions = []


        # ----------------------------------------------------
        # Search terms
        # ----------------------------------------------------

        for term in search_terms:

            position = lower_text.find(
                term.lower()
            )


            if position >= 0:

                positions.append(
                    position
                )


        # ----------------------------------------------------
        # Search numbers
        # ----------------------------------------------------

        if "number" in question_types:

            for match in re.finditer(
                r"\b\d+(?:[.,]\d+)?\b",
                compact_text
            ):

                positions.append(
                    match.start()
                )


        # ----------------------------------------------------
        # Search dates
        # ----------------------------------------------------

        if "date" in question_types:

            month_pattern = (
                r"(ianuarie|februarie|martie|"
                r"aprilie|mai|iunie|iulie|"
                r"august|septembrie|octombrie|"
                r"noiembrie|decembrie)"
            )


            for match in re.finditer(
                r"\b\d{1,2}\s+"
                + month_pattern,
                lower_text
            ):

                positions.append(
                    match.start()
                )


        # ----------------------------------------------------
        # Build local windows
        # ----------------------------------------------------

        for position in positions:

            left = max(
                0,
                position - 250
            )


            right = min(
                len(compact_text),
                position + 350
            )


            window = (
                compact_text[left:right]
                .strip()
            )


            if len(window) < 30:

                continue


            window_words = set(
                normalize_words(window)
            )


            overlap = (
                len(
                    question_words
                    &
                    window_words
                )
                /
                max(
                    len(question_words),
                    1
                )
            )


            score = (
                0.60
                * chunk["combined_score"]
                +
                0.40
                * overlap
            )


            evidence.append({

                "text":
                    window,

                "source":
                    chunk["source"],

                "page":
                    chunk["page"],

                "score":
                    score
            })


    # --------------------------------------------------------
    # Rank evidence
    # --------------------------------------------------------

    evidence.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Deduplicate windows
    # --------------------------------------------------------

    selected = []

    seen = set()


    for item in evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower()
        )


        key = (
            item["source"],
            item["page"],
            normalized[:180]
        )


        if key in seen:

            continue


        seen.add(
            key
        )


        selected.append(
            item
        )


        if len(selected) >= 8:

            break


    # --------------------------------------------------------
    # Fallback to complete chunks
    # --------------------------------------------------------

    if not selected:

        for chunk in retrieved_chunks:

            selected.append({

                "text":
                    chunk["text"],

                "source":
                    chunk["source"],

                "page":
                    chunk["page"],

                "score":
                    chunk["combined_score"]
            })


    return selected


# ============================================================
# Build LLM prompt
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


    context_text = "\n".join(
        context
    )


    return f"""
/no_think

Ești un asistent care răspunde la întrebări
despre documentele Universității Naționale
de Știință și Tehnologie POLITEHNICA București.

ÎNTREBAREA:

{question}

Folosește EXCLUSIV dovezile furnizate.

REGULI STRICTE:

- Răspunde în limba română.
- Răspunde exact la întrebarea pusă.
- Nu răspunde la o altă întrebare apropiată.
- Nu inventa informații.
- Nu folosi informații generale din afara dovezilor.
- Pentru valori numerice, folosește valoarea exactă.
- Pentru credite, verifică exact numărul de credite.
- Pentru date, verifică data de început și data de sfârșit.
- Pentru condiții, include condițiile relevante.
- Nu confunda tipurile de burse.
- Nu menționa DOVADĂ 1, DOVADĂ 2 etc.
- Nu genera o secțiune de surse.

Dacă informația necesară nu apare în dovezi, răspunde exact:

"Nu am găsit această informație în documentele disponibile."

DOVEZI:

{context_text}

RĂSPUNS:
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
            "temperature":
                0.0,

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

    retrieved_sources = [
        item["source"]
        for item in retrieved
    ]


    retrieved_pages = [
        item["page"]
        for item in retrieved
    ]


    # --------------------------------------------------------
    # Deliberately unanswerable question
    # --------------------------------------------------------

    if expected_source is None:

        return {

            "source_correct":
                True,

            "page_recall":
                True,

            "retrieval_correct":
                True
        }


    source_correct = (
        expected_source
        in retrieved_sources
    )


    page_recall = any(
        page in retrieved_pages
        for page in expected_pages
    )


    return {

        "source_correct":
            source_correct,

        "page_recall":
            page_recall,

        "retrieval_correct":
            (
                source_correct
                and
                page_recall
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
        "nu este disponibilă",
        "nu pot găsi",
    ]


    refused = any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


    # --------------------------------------------------------
    # Refusal question
    # --------------------------------------------------------

    if should_refuse:

        return {
            "answer_correct":
                refused,

            "keyword_coverage":
                1.0 if refused else 0.0,

            "matched_keywords": [],

            "missing_keywords": [],

            "refused":
                refused,

            "refusal_correct":
                refused,
        }


    # --------------------------------------------------------
    # Normal question
    # --------------------------------------------------------

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


    if expected_keywords:

        coverage = (
            len(matched)
            /
            len(expected_keywords)
        )

    else:

        coverage = 1.0


    return {
        "answer_correct":
            (
                len(missing) == 0
                and not refused
            ),

        "keyword_coverage":
            coverage,

        "matched_keywords":
            matched,

        "missing_keywords":
            missing,

        "refused":
            refused,

        "refusal_correct":
            not refused,
    }


# ============================================================
# Run evaluation
# ============================================================

results = []

evaluation_start = (
    time.perf_counter()
)


for question_data in (
    evaluation_questions
):

    question_id = (
        question_data["id"]
    )

    question = (
        question_data["question"]
    )

    category = (
        question_data["category"]
    )

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
        f"{question_id}: "
        f"{question}"
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


    for i, item in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"{i}. "
            f"{item['source']}, "
            f"pagina {item['page']} "
            f"(score "
            f"{item['combined_score']:.4f})"
        )


    retrieval_eval = (
        evaluate_retrieval(
            retrieved,
            expected_source,
            expected_pages
        )
    )


    # --------------------------------------------------------
    # Targeted evidence
    # --------------------------------------------------------

    evidence_start = (
        time.perf_counter()
    )


    evidence = (
        extract_targeted_evidence(
            question,
            retrieved
        )
    )


    evidence_time = (
        time.perf_counter()
        -
        evidence_start
    )


    print()
    print(
        "Targeted evidence:"
    )


    for i, item in enumerate(
        evidence,
        start=1
    ):

        preview = (
            item["text"]
            .replace("\n", " ")
        )


        if len(preview) > 160:

            preview = (
                preview[:160]
                + "..."
            )


        print(
            f"{i}. "
            f"{item['source']}, "
            f"pagina {item['page']} | "
            f"{preview}"
        )


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print(
        "Generating answer..."
    )


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
    # Evaluate answer
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
        if retrieval_eval[
            "retrieval_correct"
        ]
        else 0.0
    )


    answer_score = (
        answer_eval[
            "keyword_coverage"
        ]
    )


    total_score = (
        0.50 * retrieval_score
        +
        0.50 * answer_score
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

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

        "retrieval_correct":
            retrieval_eval[
                "retrieval_correct"
            ],

        "evidence_pages": [
            item["page"]
            for item in evidence
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

        "refusal_correct":
            answer_eval[
                "refusal_correct"
            ],

        "retrieval_time_seconds":
            retrieval_timing[
                "total_retrieval"
            ],

        "evidence_time_seconds":
            evidence_time,

        "llm_time_seconds":
            llm_time,

        "total_time_seconds":
            (
                retrieval_timing[
                    "total_retrieval"
                ]
                +
                evidence_time
                +
                llm_time
            ),

        "prompt_tokens":
            prompt_tokens,

        "generated_tokens":
            generated_tokens,

        "total_score":
            total_score
    })


# ============================================================
# Calculate summary
# ============================================================

evaluation_time = (
    time.perf_counter()
    -
    evaluation_start
)


total_questions = len(results)


document_retrieval_accuracy = (
    sum(
        result["source_correct"]
        for result in results
    )
    /
    total_questions
)


evidence_page_recall = (
    sum(
        result["page_recall"]
        for result in results
    )
    /
    total_questions
)


keyword_coverage = (
    sum(
        result["keyword_coverage"]
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
        result["refusal_correct"]
        for result in results
    )
    /
    total_questions
)


avg_retrieval_time = (
    sum(
        result[
            "retrieval_time_seconds"
        ]
        for result in results
    )
    /
    total_questions
)


avg_evidence_time = (
    sum(
        result[
            "evidence_time_seconds"
        ]
        for result in results
    )
    /
    total_questions
)


avg_llm_time = (
    sum(
        result[
            "llm_time_seconds"
        ]
        for result in results
    )
    /
    total_questions
)


avg_total_time = (
    sum(
        result[
            "total_time_seconds"
        ]
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
    f"{document_retrieval_accuracy:.2%}"
)


print(
    f"Evidence/page recall:      "
    f"{evidence_page_recall:.2%}"
)


print(
    f"Answer accuracy:           "
    f"{answer_accuracy:.2%}"
)


print(
    f"Keyword coverage:          "
    f"{keyword_coverage:.2%}"
)


print(
    f"Refusal accuracy:          "
    f"{refusal_accuracy:.2%}"
)


print(
    f"Average retrieval time:    "
    f"{avg_retrieval_time:.3f} s"
)


print(
    f"Average evidence time:     "
    f"{avg_evidence_time:.3f} s"
)


print(
    f"Average LLM time:          "
    f"{avg_llm_time:.3f} s"
)


print(
    f"Average total time:        "
    f"{avg_total_time:.3f} s"
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
        f"Coverage: "
        f"{result['keyword_coverage']:.0%}"
    )


# ============================================================
# Save detailed results
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
            MAX_GENERATED_TOKENS,

        "evaluation_type":
            "retrieval_and_keyword_coverage",

        "strict_keyword_evaluation":
            True
    },


    "summary": {

        "document_retrieval_accuracy":
            document_retrieval_accuracy,

        "evidence_page_recall":
            evidence_page_recall,

        "answer_accuracy":
            answer_accuracy,

        "keyword_coverage":
            keyword_coverage,

        "refusal_accuracy":
            refusal_accuracy,

        "average_retrieval_time_seconds":
            avg_retrieval_time,

        "average_evidence_time_seconds":
            avg_evidence_time,

        "average_llm_time_seconds":
            avg_llm_time,

        "average_total_time_seconds":
            avg_total_time,

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
print(
    "Evaluation complete."
)
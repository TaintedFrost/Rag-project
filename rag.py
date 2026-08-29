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

model_start = time.perf_counter()

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

model_time = (
    time.perf_counter()
    - model_start
)

print(
    f"Embedding model loaded in "
    f"{model_time:.2f} seconds."
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


# ============================================================
# Helpers
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
# Retrieve
# ============================================================

def retrieve(query):

    total_start = time.perf_counter()

    category = detect_category(query)


    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    if category is None:

        index = faiss.read_index(
            "faiss_index.bin"
        )

        mapping = list(
            range(len(chunks))
        )

    else:

        index = category_indexes[
            category
        ]

        mapping = category_mappings[
            category
        ]


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    embedding_start = time.perf_counter()

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

    embedding_time = (
        time.perf_counter()
        -
        embedding_start
    )


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    search_start = time.perf_counter()

    search_k = min(
        SEARCH_RESULTS,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    search_time = (
        time.perf_counter()
        -
        search_start
    )


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

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
            query,
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
        key=lambda x:
        x["combined_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # One chunk per page
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


    total_time = (
        time.perf_counter()
        -
        total_start
    )


    return (
        selected,
        category,
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
# Build prompt
# ============================================================

def build_prompt(
    query,
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

Răspunde la întrebarea utilizatorului folosind
EXCLUSIV informațiile din CONTEXT.

Întrebare:
{query}

Reguli stricte:

- Răspunde numai în limba română.
- Răspunde direct la întrebare.
- Citește toate contextele înainte de a răspunde.
- Include toate informațiile relevante.
- Pentru întrebări despre condiții, include toate
  valorile numerice relevante.
- Pentru întrebări despre date, include data de început
  și data de sfârșit atunci când ambele apar în context.
- Pentru întrebări despre documente, enumeră documentele
  relevante.
- Nu confunda informațiile din articole diferite.
- Nu inventa informații.
- Nu folosi cunoștințe din afara contextului.
- Nu genera surse.
- Nu menționa "CONTEXT 1", "CONTEXT 2" etc.
- Nu spune că o informație există dacă nu este prezentă.

Dacă răspunsul nu poate fi determinat din context,
scrie exact:

"Nu am găsit această informație în documentele disponibile."

CONTEXT:

{context_text}

Răspuns:
"""


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    query,
    retrieved_chunks
):

    prompt = build_prompt(
        query,
        retrieved_chunks
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
            "temperature": 0.0,
            "num_predict":
                MAX_GENERATED_TOKENS,
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
# Build sources in Python
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


        seen.add(
            key
        )


        sources.append(
            f"- {chunk['source']}, "
            f"pagina {chunk['page']}"
        )


    return sources


# ============================================================
# Main loop
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

    except (
        KeyboardInterrupt,
        EOFError
    ):

        print("\nExiting...")
        break


    if query.lower() == "exit":

        print("Exiting...")
        break


    if not query:

        continue


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
    print(
        "Se generează răspunsul..."
    )


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
            llm_time
        )


        print()
        print("=" * 70)
        print("TIMING")
        print("=" * 70)

        print(
            f"Embedding:        "
            f"{retrieval_timing['embedding']:.3f} s"
        )

        print(
            f"FAISS search:     "
            f"{retrieval_timing['faiss_search']:.3f} s"
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


    except Exception as error:

        print()
        print("=" * 70)
        print("EROARE")
        print("=" * 70)

        print(error)


    print()
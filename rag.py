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

# Large candidate pool for better reranking.
SEARCH_RESULTS = 50

# Only the best complete chunks are initially retrieved.
FINAL_RESULTS = 4

# Keep generation short.
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
        "validare",
        "înmatriculare",
        "inmatriculare",
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
# Stopwords
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
# Question intent detection
# ============================================================

def detect_question_intent(query):
    """
    Determine what the user is actually asking for.
    This is more specific than generic question-type detection.
    """

    q = query.lower()

    # --------------------------------------------------------
    # Specific numeric concepts
    # --------------------------------------------------------

    if (
        "credite" in q
        or "ects" in q
    ):
        return "credits"

    if (
        "punctaj minim" in q
        or "punctajul minim" in q
        or "puncte minim" in q
        or "număr de puncte" in q
        or "numar de puncte" in q
    ):
        return "points"

    if (
        "durata studiilor" in q
        or "durata studiilor universitare" in q
        or "cât durează studiile" in q
        or "cat dureaza studiile" in q
        or "durată" in q
        or "durata" in q
    ):
        return "duration"

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if (
        "când începe" in q
        or "cand incepe" in q
        or "când se termină" in q
        or "cand se termina" in q
        or "ce perioadă" in q
        or "ce perioada" in q
        or "care este perioada" in q
    ):
        return "dates"

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    if (
        "ce documente" in q
        or "ce document" in q
        or "ce acte" in q
        or "ce acte trebuie" in q
        or "ce trebuie depus" in q
        or "ce se depune" in q
    ):
        return "documents"

    # --------------------------------------------------------
    # Scholarship activities
    # --------------------------------------------------------

    if (
        "ce activități contribuie" in q
        or "ce activitati contribuie" in q
        or "activități contribuie" in q
        or "activitati contribuie" in q
    ):
        return "scholarship_activities"

    # --------------------------------------------------------
    # General conditions
    # --------------------------------------------------------

    if (
        "condiții" in q
        or "conditii" in q
        or "criterii" in q
        or "eligibilitate" in q
        or "cerințe" in q
        or "cerinte" in q
    ):
        return "conditions"

    return "general"


# ============================================================
# General question type detection
# ============================================================

def detect_question_types(query):

    q = query.lower()

    types = []

    if any(
        term in q
        for term in [
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
        ]
    ):
        types.append("number")

    if any(
        term in q
        for term in [
            "când",
            "cand",
            "data",
            "perioada",
            "începe",
            "incepe",
            "termină",
            "termina",
        ]
    ):
        types.append("date")

    if any(
        term in q
        for term in [
            "condiții",
            "conditii",
            "criterii",
            "eligibilitate",
            "cerințe",
            "cerinte",
        ]
    ):
        types.append("conditions")

    if any(
        term in q
        for term in [
            "documente",
            "document",
            "dosar",
            "acte",
        ]
    ):
        types.append("documents")

    return types


# ============================================================
# Text normalization
# ============================================================

def normalize_words(text):

    words = re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


# ============================================================
# Lexical score
# ============================================================

def lexical_score(query, text):

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


# ============================================================
# Phrase score
# ============================================================

def phrase_score(query, text):

    q = query.lower()
    t = text.lower()

    important_phrases = [
        "bursa de performanță",
        "bursa de performanta",
        "punctaj minim",
        "credite transferabile",
        "300 de credite",
        "număr minim de credite",
        "numar minim de credite",
        "înmatriculare",
        "inmatriculare",
        "validare",
        "bacalaureat",
        "criterii de departajare",
    ]

    score = 0.0

    for phrase in important_phrases:

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
# Load chunks
# ============================================================

print("Loading RAG data...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
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
    -
    model_start
)

print(
    f"Embedding model loaded in "
    f"{model_time:.2f} seconds."
)


# ============================================================
# Load indexes
# ============================================================

category_indexes = {}
category_mappings = {}


for category in set(
    DOCUMENT_CATEGORIES.values()
):

    index_path = (
        INDEX_FOLDER
        /
        f"{category}.index"
    )

    mapping_path = (
        INDEX_FOLDER
        /
        f"{category}_mapping.json"
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
        faiss.read_index(
            str(index_path)
        )
    )

    with mapping_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        category_mappings[category] = (
            json.load(file)
        )

# ============================================================
# Category detection
# ============================================================

def detect_category(query):

    q = query.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in q:
                return category

    return None

# ============================================================
# Retrieve
# ============================================================

def retrieve(query):

    start = time.perf_counter()

    category = detect_category(
        query
    )


    # --------------------------------------------------------
    # Choose index
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

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
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
    # FAISS
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
            query,
            chunk["text"]
        )

        phrase = phrase_score(
            query,
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
    # Keep unique pages
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
        start
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
# Targeted evidence extraction
# ============================================================

def extract_targeted_evidence(
    query,
    retrieved_chunks
):
    """
    Question-aware evidence selection.

    IMPORTANT:
    This does not ask another LLM.

    It looks for the exact concept requested by the user,
    then extracts a local context window around that concept.
    """

    start = time.perf_counter()

    intent = detect_question_intent(
        query
    )

    question_words = set(
        normalize_words(query)
    )


    evidence = []


    # --------------------------------------------------------
    # Define intent-specific patterns
    # --------------------------------------------------------

    patterns = []


    if intent == "credits":

        patterns = [
            r"credite.{0,120}ect",
            r"cel puțin.{0,120}credite",
            r"cel putin.{0,120}credite",
            r"minimum.{0,120}credite",
            r"credite.{0,100}\d+",
            r"\d+.{0,100}credite",
        ]


    elif intent == "points":

        patterns = [
            r"punctaj.{0,150}minim",
            r"punctaj.{0,100}\d+",
            r"\d+.{0,100}puncte",
            r"minimum.{0,100}puncte",
            r"minim.{0,100}puncte",
        ]


    elif intent == "duration":

        patterns = [
            r"studii.{0,100}durată.{0,100}\d+",
            r"studii.{0,100}durata.{0,100}\d+",
            r"durată.{0,100}\d+\s+ani",
            r"durata.{0,100}\d+\s+ani",
            r"\d+\s+ani.{0,100}studii",
        ]


    elif intent == "dates":

        patterns = [
            r"\b\d{1,2}\s+"
            r"(ianuarie|februarie|martie|"
            r"aprilie|mai|iunie|iulie|"
            r"august|septembrie|octombrie|"
            r"noiembrie|decembrie)",

            r"începe.{0,180}\d{1,2}",
            r"incepe.{0,180}\d{1,2}",
            r"se termină.{0,180}\d{1,2}",
            r"se termina.{0,180}\d{1,2}",
        ]


    elif intent == "documents":

        patterns = [
            r"cerere.{0,150}",
            r"curriculum vitae.{0,150}",
            r"lista de lucrări.{0,150}",
            r"diplom.{0,150}",
            r"supliment.{0,150}",
            r"dosar.{0,150}document",
        ]


    elif intent == "scholarship_activities":

        patterns = [
            r"concursuri.{0,250}",
            r"lucrări.{0,250}",
            r"articole.{0,250}",
            r"cercetare.{0,250}",
            r"invenții.{0,250}",
            r"inventii.{0,250}",
            r"sesiuni.{0,250}",
        ]


    elif intent == "conditions":

        patterns = [
            r"trebuie.{0,180}",
            r"condi.{0,180}",
            r"criter.{0,180}",
            r"eligib.{0,180}",
            r"minimum.{0,180}",
            r"minim.{0,180}",
        ]


    # --------------------------------------------------------
    # Search targeted patterns
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


        matched_windows = []


        for pattern in patterns:

            for match in re.finditer(
                pattern,
                lower_text
            ):

                start_pos = max(
                    0,
                    match.start() - 180
                )

                end_pos = min(
                    len(compact_text),
                    match.end() + 250
                )

                window = (
                    compact_text[
                        start_pos:end_pos
                    ]
                    .strip()
                )


                if len(window) >= 30:

                    matched_windows.append(
                        window
                    )


        # ----------------------------------------------------
        # If no special pattern matched, use lexical search.
        # ----------------------------------------------------

        if not matched_windows:

            for word in question_words:

                if len(word) < 4:
                    continue

                position = (
                    lower_text.find(word)
                )

                if position < 0:
                    continue

                left = max(
                    0,
                    position - 180
                )

                right = min(
                    len(compact_text),
                    position + 280
                )

                matched_windows.append(
                    compact_text[
                        left:right
                    ].strip()
                )


        # ----------------------------------------------------
        # Score windows
        # ----------------------------------------------------

        for window in matched_windows:

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


            intent_bonus = 0.0

            lower_window = (
                window.lower()
            )


            # Strong deterministic bonuses.
            if intent == "credits":

                if (
                    "credite" in lower_window
                    or
                    "ects" in lower_window
                ):
                    intent_bonus += 0.40


                if re.search(
                    r"\b\d+\b",
                    lower_window
                ):
                    intent_bonus += 0.20


            elif intent == "duration":

                if (
                    "durat" in lower_window
                    and
                    "ani" in lower_window
                ):
                    intent_bonus += 0.50


            elif intent == "dates":

                if re.search(
                    r"\b\d{1,2}\s+"
                    r"(ianuarie|februarie|martie|"
                    r"aprilie|mai|iunie|iulie|"
                    r"august|septembrie|octombrie|"
                    r"noiembrie|decembrie)",
                    lower_window
                ):
                    intent_bonus += 0.50


            elif intent == "documents":

                if any(
                    term in lower_window
                    for term in [
                        "cerere",
                        "curriculum",
                        "diplom",
                        "supliment",
                        "dosar",
                    ]
                ):
                    intent_bonus += 0.40


            elif intent == "scholarship_activities":

                if any(
                    term in lower_window
                    for term in [
                        "concurs",
                        "public",
                        "cercet",
                        "inven",
                        "sesiun",
                    ]
                ):
                    intent_bonus += 0.40


            score = (
                0.45
                * chunk["combined_score"]
                +
                0.35
                * overlap
                +
                0.20
                * intent_bonus
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
    # Sort
    # --------------------------------------------------------

    evidence.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    selected = []

    seen = set()


    for item in evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower()
        )


        # Use the actual beginning of the window as
        # the deduplication key.
        key = (
            item["source"],
            item["page"],
            normalized[:250]
        )


        if key in seen:
            continue


        seen.add(key)

        selected.append(
            item
        )


        # Keep the LLM context compact.
        if len(selected) >= 5:
            break


    # --------------------------------------------------------
    # Fallback
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


    elapsed = (
        time.perf_counter()
        -
        start
    )


    return selected, elapsed


# ============================================================
# Build LLM prompt
# ============================================================

def build_prompt(
    question,
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

Răspunde la întrebarea utilizatorului folosind
EXCLUSIV informațiile din dovezile furnizate.

ÎNTREBARE:
{question}

REGULI:
- Răspunde numai în limba română.
- Răspunde exact la întrebarea pusă.
- Nu răspunde la o altă întrebare.
- Identifică exact valoarea sau condiția cerută.
- Pentru credite, folosește numărul asociat explicit
  creditelor.
- Pentru durată, folosește durata studiilor, nu alte
  valori numerice din document.
- Pentru date, include începutul și sfârșitul perioadei
  atunci când sunt disponibile.
- Pentru documente, enumeră documentele cerute.
- Nu inventa informații.
- Nu folosi cunoștințe din afara dovezilor.
- Nu menționa DOVADĂ 1, DOVADĂ 2 etc.
- Nu genera surse.

Dacă informația necesară nu poate fi găsită în dovezi,
răspunde exact:

"Nu am găsit această informație în documentele disponibile."

DOVEZI:

{context}

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
# Main application
# ============================================================

print()
print("=" * 70)
print("ROMANIAN RAG ASSISTANT")
print("=" * 70)
print(f"LLM: {LLM_NAME}")
print(
    f"FAISS candidates: "
    f"{SEARCH_RESULTS}"
)
print(
    f"Chunks retrieved: "
    f"{FINAL_RESULTS}"
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


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    (
        retrieved_chunks,
        category,
        retrieval_timing
    ) = retrieve(query)


    # --------------------------------------------------------
    # Targeted evidence
    # --------------------------------------------------------

    (
        evidence,
        evidence_time
    ) = extract_targeted_evidence(
        query,
        retrieved_chunks
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


    print(
        f"Intenție detectată: "
        f"{detect_question_intent(query)}"
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
    print("Dovezi focalizate:")


    for i, item in enumerate(
        evidence,
        start=1
    ):

        preview = (
            item["text"]
            .replace("\n", " ")
        )


        if len(preview) > 180:

            preview = (
                preview[:180]
                + "..."
            )


        print(
            f"{i}. "
            f"{item['source']}, "
            f"pagina {item['page']} | "
            f"{preview}"
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
            evidence
        )


        print()
        print("=" * 70)
        print("RĂSPUNS")
        print("=" * 70)

        print(answer)


        print()
        print("Surse:")


        seen_sources = set()


        for chunk in retrieved_chunks:

            key = (
                chunk["source"],
                chunk["page"]
            )


            if key in seen_sources:
                continue


            seen_sources.add(
                key
            )


            print(
                f"- {chunk['source']}, "
                f"pagina {chunk['page']}"
            )


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
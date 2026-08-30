import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from ollama import chat
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LLM_NAME = "qwen3:1.7b"

METADATA_PATH = "chunk_metadata.json"
INDEX_FOLDER = Path("indexes")

# Search a large candidate pool.
SEARCH_RESULTS = 50

# Number of chunks ultimately considered by the evidence stage.
FINAL_RESULTS = 4

# Keep answers reasonably short.
MAX_GENERATED_TOKENS = 180


# ============================================================
# DOCUMENT CATEGORIES
# ============================================================

DOCUMENT_CATEGORIES = {
    "doc1.pdf": "doctorat",
    "doc2.pdf": "licenta",
    "doc3.pdf": "burse",
    "doc4.pdf": "masterat",
}


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    "doctorat": [
        "doctorat",
        "doctorand",
        "doctoranzi",
        "studii doctorale",
        "școala doctorală",
        "scoala doctorala",
        "școlile doctorale",
        "scolile doctorale",
        "admitere la doctorat",
        "admiterea la doctorat",
        "înscriere la doctorat",
        "inscriere la doctorat",
        "înscrierea la doctorat",
        "inscrierea la doctorat",
    ],

    "licenta": [
        "licență",
        "licenta",
        "studii de licență",
        "studii de licenta",
        "ciclu de licență",
        "ciclu de licenta",
        "admitere licență",
        "admitere licenta",
        "admiterea la licență",
        "admiterea la licenta",
        "concursul de admitere la licență",
        "concursul de admitere la licenta",
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
        "bursa de performanta",
        "bursa de performanță",
        "bursa de performanta",
        "bursă de merit",
        "bursa de merit",
        "bursă socială",
        "bursa sociala",
        "ajutor social",
    ],

    "masterat": [
        "masterat",
        "master",
        "masterand",
        "studii de masterat",
        "studii de master",
        "ciclu de masterat",
        "ciclu de master",
        "admitere masterat",
        "admitere master",
        "admiterea la masterat",
        "admiterea la master",
        "concursul de admitere la masterat",
        "concursul de admitere la master",
    ],
}


# ============================================================
# ROMANIAN STOPWORDS
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
    "unor",
    "unei",
    "acest",
    "aceasta",
    "aceste",
    "acestea",
}


# ============================================================
# LOAD CHUNKS
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
# LOAD EMBEDDING MODEL
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
# LOAD CATEGORY INDEXES
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
        encoding="utf-8"
    ) as file:
        category_mappings[category] = (
            json.load(file)
        )


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

def normalize_words(text):
    """
    Convert text into normalized words and remove
    common Romanian stopwords.
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


def normalize_spaces(text):
    """
    Collapse repeated whitespace.
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(query):

    query_lower = query.lower()

    # Prefer more specific phrases first.
    candidates = []

    for category, keywords in (
        CATEGORY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in query_lower:

                candidates.append(
                    (
                        len(keyword),
                        category
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# QUESTION INTENT
# ============================================================

def detect_question_intent(query):

    q = query.lower()

    # --------------------------------------------------------
    # Credits
    # --------------------------------------------------------

    if (
        "credite" in q
        or "ects" in q
    ):
        return "credits"


    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    if (
        "punctaj minim" in q
        or "punctajul minim" in q
        or "număr de puncte" in q
        or "numar de puncte" in q
        or "puncte minim" in q
    ):
        return "points"


    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if (
        "durata studiilor" in q
        or "durata studiilor universitare" in q
        or "cât durează studiile" in q
        or "cat dureaza studiile" in q
        or "cât durează" in q
        or "cat dureaza" in q
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
        or "care este perioada" in q
        or "ce perioadă" in q
        or "ce perioada" in q
        or "data de început" in q
        or "data de inceput" in q
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
# LEXICAL SCORE
# ============================================================

def lexical_score(
    query,
    text
):

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
# PHRASE SCORE
# ============================================================

def phrase_score(
    query,
    text
):

    q = query.lower()
    t = text.lower()

    phrases = [
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
        "durata studiilor",
        "studii universitare de doctorat",
        "studii universitare de masterat",
        "studii universitare de licență",
        "studii universitare de licenta",
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
# RETRIEVAL
# ============================================================

def retrieve(query):

    retrieval_start = (
        time.perf_counter()
    )

    category = detect_category(
        query
    )


    # --------------------------------------------------------
    # Choose index
    # --------------------------------------------------------

    if category is None:

        global_index_path = (
            "faiss_index.bin"
        )

        index = faiss.read_index(
            global_index_path
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
                combined,
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
        rerank_start
    )


    total_retrieval = (
        time.perf_counter()
        -
        retrieval_start
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
                total_retrieval,
        }
    )


# ============================================================
# TARGETED FACT EXTRACTION
# ============================================================

def extract_fact_evidence(
    query,
    retrieved_chunks
):
    """
    Extract evidence specifically for exact factual
    questions.

    Returns ONLY a list.
    This avoids the tuple/list mistake from the previous version.
    """

    intent = detect_question_intent(
        query
    )

    evidence = []


    # --------------------------------------------------------
    # CREDIT QUESTIONS
    # --------------------------------------------------------

    if intent == "credits":

        patterns = [
            r"cel puțin.{0,150}credite",
            r"cel putin.{0,150}credite",
            r"credite transferabile.{0,120}",
            r"\b300\b.{0,120}credite",
            r"credite.{0,120}\b300\b",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower
                )


                if match:

                    left = max(
                        0,
                        match.start() - 180
                    )

                    right = min(
                        len(text),
                        match.end() + 250
                    )


                    passage = (
                        text[left:right]
                        .strip()
                    )


                    evidence.append({

                        "text":
                            passage,

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100.0,
                    })


                    break


    # --------------------------------------------------------
    # DURATION QUESTIONS
    # --------------------------------------------------------

    elif intent == "duration":

        patterns = [
            r"studiile universitare de doctorat.{0,180}4 ani",
            r"doctorat.{0,150}4 ani",
            r"durata.{0,100}4 ani",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150
                    )

                    right = min(
                        len(text),
                        match.end() + 180
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100.0,
                    })


                    break


    # --------------------------------------------------------
    # DATE QUESTIONS
    # --------------------------------------------------------

    elif intent == "dates":

        patterns = [
            r"15 iulie.{0,300}17 iulie",
            r"începe.{0,300}15 iulie",
            r"incepe.{0,300}15 iulie",
            r"se încheie.{0,300}17 iulie",
            r"se incheie.{0,300}17 iulie",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower
                )


                if match:

                    left = max(
                        0,
                        match.start() - 180
                    )

                    right = min(
                        len(text),
                        match.end() + 300
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100.0,
                    })


                    break


    # --------------------------------------------------------
    # DOCUMENT QUESTIONS
    # --------------------------------------------------------

    elif intent == "documents":

        patterns = [
            r"cerere de înscriere.{0,500}",
            r"cerere de inscriere.{0,500}",
            r"curriculum vitae.{0,400}",
            r"lista de lucrări.{0,400}",
            r"lista de lucrari.{0,400}",
            r"diplome de absolvire.{0,400}",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150
                    )

                    right = min(
                        len(text),
                        match.end() + 450
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100.0,
                    })


                    # Do not stop at the first chunk.
                    # Document questions may need multiple
                    # pieces of the list.

                    break


    # --------------------------------------------------------
    # POINT QUESTIONS
    # --------------------------------------------------------

    elif intent == "points":

        patterns = [
            r"punctajul minim.{0,150}30",
            r"punctaj minim.{0,150}30",
            r"30 de puncte",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150
                    )

                    right = min(
                        len(text),
                        match.end() + 200
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100.0,
                    })


                    break


    # --------------------------------------------------------
    # SCHOLARSHIP ACTIVITIES
    # --------------------------------------------------------

    elif intent == "scholarship_activities":

        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            activity_terms = [
                "concursuri profesionale",
                "lucrări și articole",
                "lucrari si articole",
                "contracte de cercetare",
                "invenții și inovații",
                "inventii si inovatii",
                "sesiuni de comunicări",
                "sesiuni de comunicari",
                "alte activități deosebite",
                "alte activitati deosebite",
            ]


            hits = sum(
                term in lower
                for term in activity_terms
            )


            if hits > 0:

                evidence.append({

                    "text":
                        text,

                    "source":
                        chunk["source"],

                    "page":
                        chunk["page"],

                    "score":
                        float(hits),
                })


    # --------------------------------------------------------
    # Sort + deduplicate
    # --------------------------------------------------------

    evidence.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    selected = []

    seen = set()


    for item in evidence:

        normalized = normalize_spaces(
            item["text"]
        ).lower()


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


        # Keep prompt compact.
        if len(selected) >= 5:
            break


    return selected


# ============================================================
# FALLBACK EVIDENCE
# ============================================================

def fallback_evidence(
    retrieved_chunks
):
    """
    If deterministic extraction finds nothing,
    send the complete top chunks.
    """

    evidence = []


    for chunk in retrieved_chunks:

        evidence.append({

            "text":
                chunk["text"],

            "source":
                chunk["source"],

            "page":
                chunk["page"],

            "score":
                chunk["combined_score"],
        })


    return evidence


# ============================================================
# BUILD PROMPT
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


    intent = detect_question_intent(
        query
    )


    extra_instruction = ""


    if intent == "credits":

        extra_instruction = """
IMPORTANT:
Întrebarea este despre CREDITE.
Nu folosi durata în ani ca răspuns.
Caută explicit numărul asociat cu "credite"
sau "ECTS".
"""


    elif intent == "duration":

        extra_instruction = """
IMPORTANT:
Întrebarea este despre DURATA studiilor.
Nu răspunde cu numărul de credite.
Caută explicit durata exprimată în ani.
"""


    elif intent == "dates":

        extra_instruction = """
IMPORTANT:
Întrebarea este despre O PERIOADĂ.
Răspunsul trebuie să includă data de început
și data de sfârșit dacă acestea apar.
"""


    elif intent == "documents":

        extra_instruction = """
IMPORTANT:
Întrebarea este despre DOCUMENTE.
Enumeră documentele efectiv cerute.
"""


    return f"""
/no_think

Ești un asistent pentru documentele oficiale
ale Universității Naționale de Știință și Tehnologie
POLITEHNICA București.

ÎNTREBARE:
{query}

Răspunde EXCLUSIV folosind DOVEZILE.

REGULI:
- Răspunde în limba română.
- Răspunde exact la întrebarea pusă.
- Nu răspunde la o altă întrebare apropiată.
- Nu inventa informații.
- Nu folosi informații din afara dovezilor.
- Verifică atent valorile numerice.
- Nu confunda două valori numerice diferite.
- Nu confunda articole sau reguli diferite.
- Nu menționa DOVADĂ 1, DOVADĂ 2 etc.
- Nu genera surse.
- Nu spune că o informație există dacă nu este prezentă.

{extra_instruction}

Dacă informația necesară nu apare în dovezi,
scrie exact:

"Nu am găsit această informație în documentele disponibile."

DOVEZI:

{context}

RĂSPUNS:
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    evidence
):

    prompt = build_prompt(
        query,
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
# MAIN LOOP
# ============================================================

print()
print("=" * 70)
print("ROMANIAN RAG ASSISTANT")
print("=" * 70)

print(
    f"LLM: {LLM_NAME}"
)

print(
    f"FAISS candidates: {SEARCH_RESULTS}"
)

print(
    f"Chunks retrieved: {FINAL_RESULTS}"
)

print(
    f"Maximum generated tokens: "
    f"{MAX_GENERATED_TOKENS}"
)

print(
    "Type 'exit' to stop."
)

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
    # Fact evidence
    # --------------------------------------------------------

    evidence_start = (
        time.perf_counter()
    )


    evidence = extract_fact_evidence(
        query,
        retrieved_chunks
    )


    # --------------------------------------------------------
    # Fallback if fact extraction found nothing
    # --------------------------------------------------------

    if not evidence:

        evidence = fallback_evidence(
            retrieved_chunks
        )


    evidence_time = (
        time.perf_counter()
        -
        evidence_start
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

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

        preview = normalize_spaces(
            item["text"]
        )


        if len(preview) > 220:

            preview = (
                preview[:220]
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


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

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


    except Exception as error:

        print()
        print("=" * 70)
        print("EROARE")
        print("=" * 70)

        print(error)
        print()

        continue


    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RĂSPUNS")
    print("=" * 70)

    print(answer)


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

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


    print()
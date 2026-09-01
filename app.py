import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

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

METADATA_PATH = Path("chunk_metadata.json")
INDEX_FOLDER = Path("indexes")

SEARCH_RESULTS = 50
FINAL_RESULTS = 4
MAX_GENERATED_TOKENS = 180


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Romanian RAG Assistant",
    page_icon="🎓",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }

    .source-box {
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }

    .answer-box {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid #ddd;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🇷🇴 Romanian RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Asistent bazat pe RAG pentru documentele Universității
    Naționale de Știință și Tehnologie POLITEHNICA București
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT CATEGORIES
# ============================================================

DOCUMENT_CATEGORIES = {
    "doctorat": "doc1.pdf",
    "licenta": "doc2.pdf",
    "burse": "doc3.pdf",
    "masterat": "doc4.pdf",
}


CATEGORY_LABELS = {
    "doctorat": "Doctorat",
    "licenta": "Licență",
    "burse": "Burse",
    "masterat": "Masterat",
}


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
# STOPWORDS
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
    "acest",
    "aceasta",
    "aceste",
    "acestea",
}


# ============================================================
# QUESTION INTENT
# ============================================================

def detect_question_intent(query):

    q = query.lower()

    if "credite" in q or "ects" in q:
        return "credits"

    if (
        "punctaj minim" in q
        or "punctajul minim" in q
        or "număr de puncte" in q
        or "numar de puncte" in q
        or "puncte minim" in q
    ):
        return "points"

    if (
        "durata studiilor" in q
        or "durata studiilor universitare" in q
        or "durata doctoratului" in q
        or "durata studiilor de doctorat" in q
        or "durata studiilor universitare de doctorat" in q
        or "durata masteratului" in q
        or "durata licentei" in q
        or "durata licenței" in q
        or "cât durează studiile" in q
        or "cat dureaza studiile" in q
        or "cât durează" in q
        or "cat dureaza" in q
    ):
        return "duration"

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

    if (
        "ce documente" in q
        or "ce document" in q
        or "ce acte" in q
        or "ce acte trebuie" in q
        or "ce trebuie depus" in q
        or "ce se depune" in q
    ):
        return "documents"

    if (
        "ce activități contribuie" in q
        or "ce activitati contribuie" in q
        or "activități contribuie" in q
        or "activitati contribuie" in q
    ):
        return "scholarship_activities"

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
# CATEGORY DETECTION
# ============================================================

def detect_category(query):

    query_lower = query.lower()

    matches = []

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in query_lower:

                matches.append(
                    (
                        len(keyword),
                        category,
                    )
                )

    if not matches:
        return None

    matches.sort(reverse=True)

    return matches[0][1]


# ============================================================
# TEXT UTILITIES
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


def normalize_spaces(text):

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# CACHE DATA
# ============================================================

@st.cache_resource(show_spinner=False)
def load_resources():

    # --------------------------------------------------------
    # Chunks
    # --------------------------------------------------------

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


    # --------------------------------------------------------
    # Category indexes
    # --------------------------------------------------------

    indexes = {}
    mappings = {}


    for category in DOCUMENT_CATEGORIES:

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
                f"Lipsește indexul: {index_path}"
            )


        if not mapping_path.exists():

            raise FileNotFoundError(
                f"Lipsește mapping-ul: {mapping_path}"
            )


        indexes[category] = (
            faiss.read_index(
                str(index_path)
            )
        )


        with mapping_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            mappings[category] = (
                json.load(file)
            )


    # --------------------------------------------------------
    # Global index
    # --------------------------------------------------------

    global_index = faiss.read_index(
        "faiss_index.bin"
    )


    return (
        chunks,
        embedding_model,
        indexes,
        mappings,
        global_index,
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    (
        chunks,
        embedding_model,
        category_indexes,
        category_mappings,
        global_index,
    ) = load_resources()

except Exception as error:

    st.error(
        f"Eroare la încărcarea resurselor: {error}"
    )

    st.stop()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    category = detect_category(
        query
    )


    if category is None:

        index = global_index

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
    # Embedding
    # --------------------------------------------------------

    embedding_start = time.perf_counter()

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    embedding_time = (
        time.perf_counter()
        -
        embedding_start
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    search_start = time.perf_counter()

    search_k = min(
        SEARCH_RESULTS,
        index.ntotal,
    )


    semantic_scores, indices = (
        index.search(
            query_embedding,
            search_k,
        )
    )


    search_time = (
        time.perf_counter()
        -
        search_start
    )


    # --------------------------------------------------------
    # Candidate reranking
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    query_words = set(
        normalize_words(query)
    )


    candidates = []


    for semantic, local_index in zip(
        semantic_scores[0],
        indices[0],
    ):

        if local_index < 0:
            continue


        original_index = (
            mapping[local_index]
        )


        chunk = chunks[
            original_index
        ]


        text = chunk["text"]


        text_words = set(
            normalize_words(text)
        )


        if query_words:

            lexical = (
                len(
                    query_words
                    &
                    text_words
                )
                /
                len(query_words)
            )

        else:

            lexical = 0.0


        combined = (
            0.75 * float(semantic)
            +
            0.25 * lexical
        )


        candidates.append({

            "chunk_index":
                original_index,

            "source":
                chunk["source"],

            "page":
                chunk["page"],

            "text":
                text,

            "semantic_score":
                float(semantic),

            "keyword_score":
                lexical,

            "combined_score":
                combined,
        })


    candidates.sort(
        key=lambda x:
        x["combined_score"],
        reverse=True,
    )


    # --------------------------------------------------------
    # Unique pages
    # --------------------------------------------------------

    selected = []

    seen_pages = set()


    for candidate in candidates:

        key = (
            candidate["source"],
            candidate["page"],
        )


        if key in seen_pages:
            continue


        seen_pages.add(key)

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


    return (
        selected,
        category,
        {
            "embedding":
                embedding_time,

            "search":
                search_time,

            "reranking":
                reranking_time,
        },
    )


# ============================================================
# TARGETED EVIDENCE
# ============================================================

def extract_evidence(
    query,
    retrieved_chunks,
):

    intent = detect_question_intent(
        query
    )


    evidence = []


    # ========================================================
    # CREDITS
    # ========================================================

    if intent == "credits":

        patterns = [
            r"cel puțin.{0,150}credite",
            r"cel putin.{0,150}credite",
            r"credite transferabile.{0,150}",
            r"\b300\b.{0,150}credite",
            r"credite.{0,150}\b300\b",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower,
                )


                if match:

                    left = max(
                        0,
                        match.start() - 200,
                    )

                    right = min(
                        len(text),
                        match.end() + 300,
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100,
                    })


                    break


    # ========================================================
    # DURATION
    # ========================================================

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
                    lower,
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150,
                    )

                    right = min(
                        len(text),
                        match.end() + 200,
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100,
                    })


                    break


    # ========================================================
    # DATES
    # ========================================================

    elif intent == "dates":

        patterns = [
            r"15 iulie.{0,300}17 iulie",
            r"începe.{0,300}15 iulie",
            r"incepe.{0,300}15 iulie",
        ]


        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            lower = text.lower()


            for pattern in patterns:

                match = re.search(
                    pattern,
                    lower,
                )


                if match:

                    left = max(
                        0,
                        match.start() - 180,
                    )

                    right = min(
                        len(text),
                        match.end() + 300,
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100,
                    })


                    break


    # ========================================================
    # DOCUMENTS
    # ========================================================

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
                    lower,
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150,
                    )

                    right = min(
                        len(text),
                        match.end() + 450,
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100,
                    })


                    break


    # ========================================================
    # POINTS
    # ========================================================

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
                    lower,
                )


                if match:

                    left = max(
                        0,
                        match.start() - 150,
                    )

                    right = min(
                        len(text),
                        match.end() + 200,
                    )


                    evidence.append({

                        "text":
                            text[left:right],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "score":
                            100,
                    })


                    break


    # ========================================================
    # SCHOLARSHIP ACTIVITIES
    # ========================================================

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
                        hits,
                })


    # ========================================================
    # GENERAL / CONDITIONS
    # ========================================================

    else:

        for chunk in retrieved_chunks:

            text = normalize_spaces(
                chunk["text"]
            )

            evidence.append({

                "text":
                    text,

                "source":
                    chunk["source"],

                "page":
                    chunk["page"],

                "score":
                    chunk["combined_score"],
            })


    # ========================================================
    # Sort + deduplicate
    # ========================================================

    evidence.sort(
        key=lambda x:
        x["score"],
        reverse=True,
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
            normalized[:250],
        )


        if key in seen:
            continue


        seen.add(key)

        selected.append(
            item
        )


        if len(selected) >= 4:
            break


    # --------------------------------------------------------
    # Fallback to chunks
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
                    chunk["combined_score"],
            })


    return selected


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    query,
    evidence,
):

    intent = detect_question_intent(
        query
    )


    context_parts = []


    for i, item in enumerate(
        evidence,
        start=1,
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


    special_instruction = ""


    if intent == "credits":

        special_instruction = """
ÎNTREBAREA ESTE DESPRE CREDITE.
Răspunsul trebuie să folosească numărul
care apare explicit lângă "credite" sau "ECTS".
NU folosi numărul de ani ca răspuns.
"""


    elif intent == "duration":

        special_instruction = """
ÎNTREBAREA ESTE DESPRE DURATA STUDIILOR.
Răspunsul trebuie să folosească durata exprimată
în ani. NU folosi numărul de credite.
"""


    elif intent == "dates":

        special_instruction = """
ÎNTREBAREA ESTE DESPRE O PERIOADĂ.
Precizează data și ora de început și data și ora
de sfârșit dacă acestea sunt prezente.
"""


    elif intent == "documents":

        special_instruction = """
ÎNTREBAREA ESTE DESPRE DOCUMENTE.
Enumeră documentele cerute în dovezi.
"""


    return f"""
/no_think

Ești un asistent pentru documentele oficiale
ale Universității Naționale de Știință și Tehnologie
POLITEHNICA București.

ÎNTREBARE:
{query}

Răspunde numai pe baza dovezilor de mai jos.

REGULI:
- Răspunde în limba română.
- Răspunde direct și concis.
- Nu răspunde la o întrebare diferită.
- Nu inventa informații.
- Nu folosi cunoștințe din afara dovezilor.
- Verifică valorile numerice cu atenție.
- Nu confunda două valori numerice diferite.
- Nu confunda tipurile de burse.
- Nu menționa DOVADĂ 1, DOVADĂ 2 etc.
- Nu crea o secțiune de surse.
- Dacă informația nu apare în dovezi, răspunde exact:

"Nu am găsit această informație în documentele disponibile."

{special_instruction}

DOVEZI:

{context}

RĂSPUNS:
"""


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    query,
    evidence,
):

    prompt = build_prompt(
        query,
        evidence,
    )


    start = time.perf_counter()


    response = chat(
        model=LLM_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        think=False,
        options={
            "temperature": 0.0,
            "num_predict":
                MAX_GENERATED_TOKENS,
        },
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    answer = (
        response.message.content
        .strip()
    )


    prompt_tokens = getattr(
        response,
        "prompt_eval_count",
        None,
    )


    generated_tokens = getattr(
        response,
        "eval_count",
        None,
    )


    return (
        answer,
        elapsed,
        prompt_tokens,
        generated_tokens,
    )


# ============================================================
# STREAMLIT INPUT
# ============================================================

question = st.text_area(
    "Întrebarea ta",
    placeholder=(
        "Exemplu: Care este durata studiilor "
        "universitare de doctorat?"
    ),
    height=100,
)


ask_button = st.button(
    "🔎 Întreabă",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Introdu o întrebare."
        )

        st.stop()


    question = question.strip()


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = (
        time.perf_counter()
    )


    with st.spinner(
        "Caut informațiile relevante..."
    ):

        (
            retrieved_chunks,
            category,
            retrieval_times,
        ) = retrieve(
            question
        )


        evidence = extract_evidence(
            question,
            retrieved_chunks,
        )


    total_retrieval_time = (
        time.perf_counter()
        -
        retrieval_start
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    with st.spinner(
        "Generez răspunsul..."
    ):

        try:

            (
                answer,
                llm_time,
                prompt_tokens,
                generated_tokens,
            ) = generate_answer(
                question,
                evidence,
            )

        except Exception as error:

            st.error(
                "Eroare la comunicarea cu Ollama: "
                f"{error}"
            )

            st.stop()


    # ========================================================
    # ANSWER
    # ========================================================


    st.markdown("### 💬 Răspuns")

    with st.container(border=True):
        st.markdown(answer)


    # ========================================================
    # TOPIC
    # ========================================================

    st.markdown(
        "### 📚 Detalii căutare"
    )


    col1, col2 = st.columns(2)


    with col1:

        if category:

            st.info(
                f"**Categorie:** "
                f"{CATEGORY_LABELS.get(category, category)}"
            )

        else:

            st.info(
                "**Categorie:** General"
            )


    with col2:

        intent_labels = {
            "credits": "Credite",
            "points": "Punctaj",
            "duration": "Durată",
            "dates": "Perioadă / date",
            "documents": "Documente",
            "scholarship_activities": "Activități pentru bursă",
            "conditions": "Condiții",
            "general": "General",
        }

        intent = detect_question_intent(question)

        st.info(
            f"**Intenție:** "
            f"{intent_labels.get(intent, 'General')}"
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.markdown(
        "### 📖 Surse"
    )


    seen_sources = set()

    for item in evidence:

        key = (
            item["source"],
            item["page"],
        )

        if key in seen_sources:
            continue

        seen_sources.add(key)

        st.markdown(
            f"""
            <div class="source-box">
            📄 <strong>{item['source']}</strong>
            — pagina <strong>{item['page']}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # OPTIONAL EVIDENCE
    # ========================================================

    with st.expander(
        "🔍 Vezi dovezile folosite"
    ):

        for i, item in enumerate(
            evidence,
            start=1,
        ):

            st.markdown(
                f"**Dovada {i} — "
                f"{item['source']}, "
                f"pagina {item['page']}**"
            )

            st.write(
                item["text"]
            )

            st.divider()


    # ========================================================
    # TIMING
    # ========================================================

    total_time = (
        total_retrieval_time
        +
        llm_time
    )


    with st.expander(
        "⏱️ Performanță"
    ):

        st.write(
            f"Embedding: "
            f"{retrieval_times['embedding']:.3f} s"
        )

        st.write(
            f"FAISS search: "
            f"{retrieval_times['search']:.3f} s"
        )

        st.write(
            f"Reranking: "
            f"{retrieval_times['reranking']:.3f} s"
        )

        st.write(
            f"Total retrieval: "
            f"{total_retrieval_time:.3f} s"
        )

        st.write(
            f"LLM generation: "
            f"{llm_time:.3f} s"
        )

        st.write(
            f"Total response: "
            f"{total_time:.3f} s"
        )

        if prompt_tokens is not None:

            st.write(
                f"Prompt tokens: "
                f"{prompt_tokens}"
            )

        if generated_tokens is not None:

            st.write(
                f"Generated tokens: "
                f"{generated_tokens}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Local RAG system • "
    "FAISS • "
    "sentence-transformers • "
    "Qwen3 1.7B • "
    "Ollama"
)
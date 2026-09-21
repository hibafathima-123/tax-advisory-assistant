import os
import re
from math import inf
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only when faiss-cpu is missing
    faiss = None


GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Local embedding model used to turn chunks/queries into vectors for
# FAISS similarity search. Runs on CPU, no API key required.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_TOP_K = 6
FOREIGN_CURRENCY_CODES = {
    "aed",
    "aud",
    "cad",
    "chf",
    "cny",
    "eur",
    "gbp",
    "hkd",
    "jpy",
    "kwd",
    "omr",
    "qar",
    "sar",
    "sgd",
    "usd",
}
INR_PATTERN = re.compile(r"₹|\b(?:rs\.?|inr|rupees?|indian rupees?)\b", re.IGNORECASE)
LAKH_CRORE_MULTIPLIERS = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
}
FOREIGN_CURRENCY_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(FOREIGN_CURRENCY_CODES)) + r")\b",
    re.IGNORECASE,
)
TEXT_FILE_ENCODINGS = ("utf-8", "utf-8-sig")


def load_groq_client():
    """Create a Groq client using the OpenAI-compatible API."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        raise ValueError(
            "GROQ_API_KEY not found. Add your Groq API key to the .env file "
            "before launching the app."
        )

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_API_BASE,
    )


def load_documents(data_file_path: str):
    """Load the tax data and split it into retrieval-friendly chunks."""
    data_path = Path(data_file_path).expanduser().resolve()
    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {data_path}")

    last_error = None
    text = None
    for encoding in TEXT_FILE_ENCODINGS:
        try:
            text = data_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if text is None:
        raise UnicodeError(
            f"Could not decode {data_path} using {', '.join(TEXT_FILE_ENCODINGS)}"
        ) from last_error

    documents = [Document(page_content=text, metadata={"source": str(data_path)})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} document(s) and split into {len(chunks)} chunks")
    return chunks


class FaissRetriever:
    """Local semantic retriever: Sentence-Transformer embeddings + FAISS IndexFlatIP.

    Pipeline:
        chunks -> SentenceTransformer embeddings (L2-normalized)
               -> faiss.IndexFlatIP (inner product == cosine similarity on
                  normalized vectors)
               -> top-k nearest chunks for a query embedding

    Exposes the same `.invoke(query) -> list[Document]` interface the old
    TFIDFRetriever had, so nothing downstream (get_answer, app.py caching)
    needs to change.
    """

    def __init__(self, documents, embedding_model_name: str = EMBEDDING_MODEL_NAME, k: int = RETRIEVER_TOP_K):
        if faiss is None:
            raise ImportError(
                "The 'faiss' package is not installed. Install it with:\n"
                "    pip install faiss-cpu\n"
                "then restart the app."
            )

        if not documents:
            raise ValueError(
                "No chunks were produced from the knowledge base — tax_data.txt "
                "appears to be empty, so there is nothing to index."
            )

        self.documents = documents
        self.k = k

        try:
            self.model = SentenceTransformer(embedding_model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the embedding model '{embedding_model_name}'. "
                "Check your internet connection (first run downloads the model "
                "from Hugging Face) and that 'sentence-transformers' is installed "
                f"correctly.\nOriginal error: {exc}"
            ) from exc

        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
            show_progress_bar=False,
        ).astype("float32")
        embeddings = np.ascontiguousarray(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def invoke(self, query: str):
        """Return the top-k most semantically similar chunks for `query`."""
        if not query or not query.strip():
            return []

        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        query_embedding = np.ascontiguousarray(query_embedding)

        top_k = min(self.k, self.index.ntotal)
        _scores, indices = self.index.search(query_embedding, top_k)

        # FAISS pads with -1 when fewer than top_k vectors are in the index;
        # filter those out so an under-filled index never raises IndexError.
        return [self.documents[i] for i in indices[0] if i != -1]


def build_retriever(documents):
    """Build a local FAISS + Sentence-Transformer semantic retriever (TF-IDF removed)."""
    print(f"Building FAISS retriever using {EMBEDDING_MODEL_NAME} ...")
    retriever = FaissRetriever(documents, embedding_model_name=EMBEDDING_MODEL_NAME, k=RETRIEVER_TOP_K)
    print(f"FAISS index ready — {retriever.index.ntotal} chunks indexed (dim={retriever.index.d})")
    return retriever


def extract_age(query: str) -> int | None:
    """Extract age when the user explicitly mentions it."""
    age_match = re.search(r"\b(\d{1,3})\s*years?\s*old\b", query.lower())
    return int(age_match.group(1)) if age_match else None


def _strip_age_mentions(query: str) -> str:
    """Remove 'NN years old' phrases so age is never mistaken for a money amount."""
    return re.sub(r"\b\d{1,3}\s*years?\s*old\b", " ", query, flags=re.IGNORECASE)


def extract_numeric_amounts(query: str) -> list[int]:
    """Extract raw numeric amounts from a query, ignoring the person's stated age."""
    cleaned = _strip_age_mentions(query)
    return [
        int(match.replace(",", ""))
        for match in re.findall(r"\b\d[\d,]*\b", cleaned)
        if match.replace(",", "").isdigit()
    ]


def extract_inr_amount(query: str) -> int | None:
    """Extract an amount explicitly labeled in INR/Rs/₹, including lakh/crore phrasing."""
    cleaned = _strip_age_mentions(query)
    matches = []

    currency = r"(?:₹|rs\.?|inr|rupees?|indian rupees?)"
    unit = r"(lakhs?|lacs?|crores?)"

    # "₹15 lakh", "Rs 15 lakh", "15 lakh rupees", "INR 1,50,000", "1,50,000 rs"
    pattern = re.compile(
        rf"{currency}\s*(\d[\d,]*(?:\.\d+)?)\s*{unit}?"
        rf"|(\d[\d,]*(?:\.\d+)?)\s*{unit}?\s*{currency}",
        re.IGNORECASE,
    )
    for match in pattern.finditer(cleaned):
        number = match.group(1) or match.group(3)
        unit_word = (match.group(2) or match.group(4) or "").lower()
        if not number:
            continue
        value = float(number.replace(",", ""))
        multiplier = LAKH_CRORE_MULTIPLIERS.get(unit_word, 1)
        matches.append(int(round(value * multiplier)))

    if matches:
        return max(matches)

    # Fall back to a bare "15 lakh" / "2 crore" mention with no currency marker at all.
    bare_pattern = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*{unit}", re.IGNORECASE)
    for match in bare_pattern.finditer(cleaned):
        value = float(match.group(1).replace(",", ""))
        multiplier = LAKH_CRORE_MULTIPLIERS.get(match.group(2).lower(), 1)
        matches.append(int(round(value * multiplier)))

    return max(matches) if matches else None


def mentions_foreign_currency(query: str) -> bool:
    """Detect common foreign currency codes in the query."""
    return bool(FOREIGN_CURRENCY_PATTERN.search(query))


def detect_taxable_income_amount(query: str) -> int | None:
    """Infer the taxable INR amount when the query provides one."""
    inr_amount = extract_inr_amount(query)
    if inr_amount is not None:
        return inr_amount

    if mentions_foreign_currency(query):
        return None

    numeric_amounts = extract_numeric_amounts(query)
    return max(numeric_amounts) if numeric_amounts else None


def is_tax_calculation_query(query: str) -> bool:
    """Check whether the user is asking for a tax calculation."""
    normalized = query.lower()

    calculation_phrases = (
        "calculate income tax",
        "calculate tax",
        "tax calculation",
        "calculate my tax",
        "how much tax",
        "what tax",
        "tax should i pay",
        "do i need to pay tax",
        "should i pay tax",
        "tax liability",
        "need to pay tax",
        "tax on an income",
        "tax for an income",
    )

    return "tax" in normalized and any(
        phrase in normalized for phrase in calculation_phrases
    )

def format_inr(amount: int | float) -> str:
    """Format a number in Indian currency notation."""
    rounded_amount = int(round(amount))
    sign = "-" if rounded_amount < 0 else ""
    digits = str(abs(rounded_amount))

    if len(digits) <= 3:
        formatted = digits
    else:
        last_three = digits[-3:]
        remaining = digits[:-3]
        parts = []
        while len(remaining) > 2:
            parts.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            parts.append(remaining)
        formatted = ",".join(reversed(parts)) + "," + last_three

    return f"Rs. {sign}{formatted}"


def calculate_progressive_tax(taxable_income: int, slabs: list[tuple[int, float]]) -> float:
    """Calculate tax for slab tuples of (upper_limit, rate)."""
    tax = 0.0
    lower_limit = 0

    for upper_limit, rate in slabs:
        if taxable_income <= lower_limit:
            break

        taxable_portion = min(taxable_income, upper_limit) - lower_limit
        if taxable_portion > 0:
            tax += taxable_portion * rate

        lower_limit = upper_limit

    return tax


def calculate_old_regime_tax(taxable_income: int, age: int | None = None) -> float:
    """Estimate old-regime tax including cess, based on the knowledge base slabs."""
    if age is not None and age >= 80:
        slabs = [(500000, 0.0), (1000000, 0.20), (inf, 0.30)]
    elif age is not None and age >= 60:
        slabs = [(300000, 0.0), (500000, 0.05), (1000000, 0.20), (inf, 0.30)]
    else:
        slabs = [(250000, 0.0), (500000, 0.05), (1000000, 0.20), (inf, 0.30)]

    base_tax = calculate_progressive_tax(taxable_income, slabs)
    if taxable_income <= 500000:
        base_tax = max(0.0, base_tax - min(base_tax, 12500.0))

    return round(base_tax * 1.04)


def calculate_new_regime_tax(taxable_income: int) -> float:
    """Estimate new-regime tax including cess, based on the knowledge base slabs."""
    slabs = [
        (400000, 0.0),
        (800000, 0.05),
        (1200000, 0.10),
        (1600000, 0.15),
        (2000000, 0.20),
        (2400000, 0.25),
        (inf, 0.30),
    ]
    base_tax = calculate_progressive_tax(taxable_income, slabs)
    if taxable_income <= 1200000:
        base_tax = 0.0

    return round(base_tax * 1.04)


def build_tax_guidance(query: str) -> dict | None:
    """Provide a deterministic answer for common tax-estimation questions."""
    if not is_tax_calculation_query(query):
        return None

    age = extract_age(query)
    taxable_income = detect_taxable_income_amount(query)
    foreign_currency = mentions_foreign_currency(query)
    numeric_amounts = extract_numeric_amounts(query)

    if foreign_currency and taxable_income is None and numeric_amounts:
        foreign_amount = format(numeric_amounts[0], ",")
        return {
            "answer": (
                f"I can estimate income tax only from the taxable amount in INR. "
                f"Your question mentions {foreign_amount} in a foreign currency, but it does not include the INR amount after conversion, so I cannot calculate the exact tax yet.\n\n"
                f"Once you share the taxable income in INR, I can estimate it. Based on the knowledge base, tax becomes zero up to {format_inr(500000)} under the old regime because of the Section 87A rebate, and zero up to {format_inr(1200000)} under the new regime because of the rebate."
            ),
            "retrieval_query": (
                "income tax slabs section 87A rebate old regime new regime "
                "taxable income in INR"
            ),
        }

    if taxable_income is None:
        return None

    old_regime_tax = calculate_old_regime_tax(taxable_income, age=age)
    new_regime_tax = calculate_new_regime_tax(taxable_income)
    income_text = format_inr(taxable_income)

    if old_regime_tax == 0 and new_regime_tax == 0:
        answer = (
            f"Assuming {income_text} is your taxable income in INR, your estimated income tax is {format_inr(0)} under both the old regime and the new regime.\n\n"
            f"Under the old regime, the Section 87A rebate keeps tax at zero up to {format_inr(500000)}. Under the new regime, the Section 87A rebate keeps tax at zero up to {format_inr(1200000)}."
        )
    else:
        answer = (
            f"Assuming {income_text} is your taxable income in INR, your estimated tax is {format_inr(old_regime_tax)} under the old regime and {format_inr(new_regime_tax)} under the new regime.\n\n"
            f"These estimates include 4% health and education cess, and they do not factor in surcharge or extra deductions/exemptions unless you have mentioned them."
        )

    if age is not None and age >= 60:
        answer += " The old-regime estimate uses the age-based slab applicable to senior citizens."

    return {
        "answer": answer,
        "retrieval_query": query,
    }


def enrich_query(query: str) -> str:
    """Expand user phrasing into tax terms that exist in the knowledge base."""
    normalized = query.lower()
    hints = []

    age = extract_age(query)
    if age is not None:
        if age >= 80:
            hints.append("super senior citizen aged 80 years and above")
        elif age >= 60:
            hints.append("senior citizen aged 60 to 79 years")

    amount = detect_taxable_income_amount(query)
    if amount is not None:
        if amount >= 1000:
            hints.append(f"taxable income around Rs. {amount:,}")
            if amount <= 500000:
                hints.append("rebate under section 87A old regime zero tax up to Rs. 5,00,000")
            if amount <= 1200000:
                hints.append("new regime zero tax up to Rs. 12,00,000")

    if is_tax_calculation_query(query):
        hints.append("income tax slabs tax liability old regime new regime section 87A rebate")

    if not hints:
        return query

    return f"{query}\n\nRelated tax terms: " + "; ".join(hints)


def build_prompt(context: str, query: str) -> str:
    """Create a grounded answer prompt for the Groq model."""
    return f"""You are a helpful and knowledgeable Indian Tax Advisory Assistant.
Use only the information provided in the context below to answer the user's question.
If the answer is not found in the context, say: "I don't have enough information about that in my knowledge base. Please consult a tax professional."

Be concise, clear, and accurate. Format numbers with Indian notation (for example, Rs. 1,50,000).
If the context clearly establishes whether the tax is zero or payable, answer that directly in the first sentence.
When useful, mention the old regime and new regime separately.
Do not mix old-regime rules and new-regime rules in the same explanation.
Do not say a special age-based slab applies under the new regime unless the context explicitly says that.

Context:
{context}

Question: {query}

Answer:"""


def get_answer(rag_state, query: str) -> dict:
    """Run a user query through the retriever and Groq chat completion API."""
    if not query or not query.strip():
        return {
            "answer": "Please enter a valid question.",
            "sources": [],
        }

    try:
        tax_guidance = build_tax_guidance(query)
        retrieval_query = enrich_query(
            tax_guidance["retrieval_query"] if tax_guidance else query
        )
        source_docs = rag_state["retriever"].invoke(retrieval_query)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        if tax_guidance:
            return {
                "answer": tax_guidance["answer"],
                "sources": [doc.page_content[:250] + "..." for doc in source_docs],
            }

        if not source_docs:
            return {
                "answer": (
                    "I couldn't find any relevant information in the knowledge "
                    "base for that question. Please try rephrasing it, or "
                    "consult a tax professional."
                ),
                "sources": [],
            }

        prompt = build_prompt(context, query)

        response = rag_state["client"].chat.completions.create(
            model=rag_state["model_name"],
            temperature=0.1,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content or "No answer generated."
        sources = [doc.page_content[:250] + "..." for doc in source_docs]

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
    except Exception as exc:
        return {
            "answer": f"Error generating answer: {exc}",
            "sources": [],
        }


def initialize_rag(data_file_path: str):
    """Load the Groq client and the local retriever."""
    client = load_groq_client()
    documents = load_documents(data_file_path)
    retriever = build_retriever(documents)

    return {
        "client": client,
        "model_name": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "retriever": retriever,
    }
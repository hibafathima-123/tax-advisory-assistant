💼 AI-Powered Tax Advisory Assistant
Final Capstone Project — Data Science
Technology: RAG (Retrieval-Augmented Generation) + Deterministic Tax Calculation
---
📌 Project Overview
This project builds an AI-powered Tax Advisory Chatbot for Indian income tax questions. It combines two things:
Retrieval-Augmented Generation (RAG) — for general/explanatory tax questions, the app retrieves relevant passages from a curated knowledge file and asks a Groq-hosted LLM to answer using only that retrieved context.
A deterministic Python tax calculator — for questions asking "how much tax will I pay", the app skips the LLM entirely and computes the exact old-regime / new-regime tax (with cess and the Section 87A rebate) using fixed slab formulas, so numeric answers are never left to LLM guesswork.
Retrieval in this project is **semantic, embedding-based** — a local `SentenceTransformer` (`sentence-transformers/all-MiniLM-L6-v2`) embeds both the knowledge-base chunks and the user's query, and a **FAISS** `IndexFlatIP` index (using L2-normalized embeddings, so inner product = cosine similarity) finds the top-6 most similar chunks. There is no TF-IDF, no keyword-frequency matching, and no scikit-learn dependency in this implementation — retrieval is based on semantic meaning, not literal term overlap.
---
🧭 How It Works
```text
                         User Query
                              │
                              ▼
                    Is this a tax-CALCULATION
                    question? (regex intent check)
                              │
              ┌───────────────┴───────────────┐
              │ Yes                            │ No
              ▼                                ▼
   Python computes old-regime /        Embed the query, then retrieve
   new-regime tax directly             top-6 chunks via FAISS similarity
   (slabs + 4% cess + 87A rebate)      search over tax_data.txt
              │                                  │
              │                                  ▼
              │                       Build a context-only prompt
              │                       and call the Groq LLM
              │                                  │
              └───────────────┬──────────────────┘
                               ▼
                 Answer + retrieved source chunks
                    rendered back in the UI
```
Retrieval always runs — even on the calculator path — so the retrieved chunks can still be shown to the user under "View Source Context."
---
🗂️ Project Structure
```text
tax_advisory_assistant/
│
├── app.py                          ← Streamlit UI (frontend)
├── rag_core.py                     ← RAG + calculator logic (backend)
├── tax_data.txt  (or tax_data_.txt)← Knowledge base (Indian tax data);
│                                      app.py uses whichever file exists
├── requirements.txt                ← Python dependencies
├── .env                            ← Your local GROQ_API_KEY (create this yourself, not committed)
└── README.md                       ← This file
```
> There is no `.env.example` file in this project yet — you'll create your own `.env` file in Step 6 below.
---
⚙️ Tech Stack
Component	Technology
UI Framework	Streamlit
RAG Framework	LangChain (`RecursiveCharacterTextSplitter`, `Document`)
Embeddings	`sentence-transformers/all-MiniLM-L6-v2` — local, CPU, runs via the `sentence-transformers` library
Retrieval	FAISS — `faiss.IndexFlatIP` (k=6) over L2-normalized embeddings, so inner product = cosine similarity. No keyword/TF-IDF matching.
Deterministic Calculator	Plain Python — regex-based intent/amount/age extraction + progressive tax-slab math
LLM Provider	Groq — accessed via the standard `openai` Python SDK pointed at Groq's OpenAI-compatible endpoint
Default Model	`openai/gpt-oss-120b` (override with the `GROQ_MODEL` environment variable — no code changes needed)
Config	`python-dotenv` (loads `GROQ_API_KEY` from `.env`)
Language	Python 3.10+
---
🚀 Step-by-Step Setup in VS Code
STEP 1: Prerequisites
Make sure you have these installed:
Python 3.10 or higher → https://www.python.org/downloads/
VS Code → https://code.visualstudio.com/
Git (optional) → https://git-scm.com/
Check Python version:
```bash
python --version
```
---
STEP 2: Create the Project Folder
Open VS Code, then open a terminal (`Ctrl + `` ` ``):
```bash
mkdir tax_advisory_assistant
cd tax_advisory_assistant
```
---
STEP 3: Copy All Project Files
Copy these files into the `tax_advisory_assistant/` folder:
`app.py`
`rag_core.py`
`tax_data.txt` (your knowledge base — `tax_data_.txt` also works as a fallback filename)
`requirements.txt`
Your folder should look like:
```text
tax_advisory_assistant/
├── app.py
├── rag_core.py
├── tax_data.txt
└── requirements.txt
```
---
STEP 4: Create a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it:

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```
You should see `(venv)` appear at the start of your terminal prompt.
---
STEP 5: Install Dependencies
```bash
pip install -r requirements.txt
```
This installs LangChain, the OpenAI SDK (used to talk to Groq), Streamlit, `sentence-transformers` (which pulls in PyTorch CPU automatically for embeddings), `faiss-cpu` (for vector similarity search), and a few small supporting libraries. There's no scikit-learn in this project anymore.

> **First run note:** `sentence-transformers` downloads the MiniLM embedding model (~80 MB) from Hugging Face the first time the app runs, and caches it locally afterward — so the very first startup needs an internet connection, but every run after that works offline.
---
STEP 6: Set Up Your Groq API Key
This project uses the Groq API (via the OpenAI-compatible SDK), not the OpenAI API directly.
1. Get your API key
Visit https://console.groq.com/keys and create a free API key.
2. Create a `.env` file
In the project root, create a file named `.env` (there's no template file to copy — just create it):
```env
GROQ_API_KEY=your-groq-api-key
```
3. (Optional) Choose a different model
By default the app uses `openai/gpt-oss-120b`. To use a different Groq-hosted model, set an environment variable — no code editing required:
```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.1-70b-versatile
```
Check https://console.groq.com/docs/models for the current list of available Groq models — model availability changes over time, so confirm a model name is still active before setting it.
---
STEP 7: Run the Application
```bash
streamlit run app.py
```
You should see output like:
```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```
The browser will open automatically at `http://localhost:8501`.
---
STEP 8: First Run Behavior
On first run, the app will:
Read `tax_data.txt` (or `tax_data_.txt`) and split it into chunks
Load the `all-MiniLM-L6-v2` embedding model and embed every chunk, then build a FAISS `IndexFlatIP` index over those embeddings (downloads the model from Hugging Face the very first time only)
Connect to the Groq API using your `.env` key
Show:
```text
   ✅ Knowledge Base Ready
   ```
Thanks to `@st.cache_resource`, this initialization runs only once per app session — subsequent questions reuse the cached retriever and client.
---
🧪 Sample Test Questions
These match the clickable sample-question buttons in the app's sidebar:
#	Question	Answered By
1	What are the income tax slabs for FY 2026-27?	LLM + retrieved context
2	What is Section 80C deduction limit?	LLM + retrieved context
3	What is the difference between old and new tax regime?	LLM + retrieved context
4	How much can I save under Section 80D?	LLM + retrieved context
5	What is the tax rebate under Section 87A?	LLM + retrieved context
6	How is HRA exemption calculated?	LLM + retrieved context
7	When is the last date to file ITR?	LLM + retrieved context
8	What is TDS and how does it work?	LLM + retrieved context
9	Are capital gains taxable?	LLM + retrieved context
10	What is standard deduction for salaried employees?	LLM + retrieved context
You can also try a calculation question, which is routed to the deterministic calculator instead of the LLM, e.g.:
> "How much tax will I pay on Rs. 8,00,000 income?"
> "My annual income is ₹15 lakh and I am 30 years old, how much tax will I pay?"

Both `₹` and spelled-out "Rs./INR/rupees" work, and "lakh"/"crore" amounts are converted automatically. A stated age is extracted separately and never confused with the income figure.

You can also try a paraphrased question that shares few exact keywords with the knowledge base, e.g. "Can you explain the tax relief available under Section 87A?" — semantic (embedding-based) retrieval is designed to still surface the right chunk here, which a purely keyword-based retriever could miss.
---
📊 Example Output
Question:
What is Section 80C deduction limit?
Answer:
> The answer is generated from whatever `tax_data.txt` actually contains, phrased according to the retrieved context. The exact wording will depend on your knowledge-base content — this project doesn't hardcode any tax facts into the answer text itself.
Question:
How much tax will I pay on Rs. 8,00,000 income?
Answer:
> Handled entirely by `build_tax_guidance()` in `rag_core.py` — no LLM call is made. The response states the estimated old-regime and new-regime tax (including 4% cess), based on the fixed slab tables in the code.
---
🧠 How the Pipeline Actually Works (For Viva)
1. Document Loading
```python
data_path = Path(data_file_path).expanduser().resolve()
for encoding in ("utf-8", "utf-8-sig"):
    try:
        text = data_path.read_text(encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
documents = [Document(page_content=text, metadata={"source": str(data_path)})]
```
Reads the knowledge file directly, trying `utf-8` first and falling back to `utf-8-sig` if that fails, then wraps it as a LangChain `Document`. No `TextLoader` class is used.
---
2. Text Chunking
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)
chunks = splitter.split_documents(documents)
```
Splits the document into overlapping chunks for retrieval.
---
3. Retrieval Index (Sentence-Transformer embeddings + FAISS)
```python
class FaissRetriever:
    def __init__(self, documents, embedding_model_name=EMBEDDING_MODEL_NAME, k=RETRIEVER_TOP_K):
        self.model = SentenceTransformer(embedding_model_name)
        embeddings = self.model.encode(
            [doc.page_content for doc in documents],
            convert_to_numpy=True,
            normalize_embeddings=True,   # inner product == cosine similarity
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
```
Each chunk is embedded once with `sentence-transformers/all-MiniLM-L6-v2`, normalized, and added to a FAISS `IndexFlatIP` index. At query time, the query is embedded the same way and `index.search()` returns the indices of the top-6 nearest chunks by cosine similarity — retrieval is based on semantic meaning, not term overlap. The class also guards against a few real-world failure cases: a missing `faiss-cpu` install, an empty knowledge base, and an embedding-model load failure all raise clear, specific errors instead of crashing silently.
---
4. Query Enrichment
```python
retrieval_query = enrich_query(query)
```
Before retrieval, `enrich_query()` appends helpful hint terms detected in the question — e.g. an age band, a taxable-income figure, or "section 87A rebate" — so the FAISS search has more relevant text to embed and match against.
---
5. Intent Check — Calculation vs. Explanation
```python
tax_guidance = build_tax_guidance(query)
```
`is_tax_calculation_query()` checks for phrases like "how much tax" or "tax liability." If matched, `build_tax_guidance()` extracts the taxable income (and age, if given) and computes the tax directly — the LLM is never called for this branch.

**Amount & age extraction.** Taxable income is pulled out with `extract_inr_amount()`, which understands the `₹` symbol, the words "Rs./INR/rupees", and "lakh"/"crore" multipliers (so "₹15 lakh" resolves to Rs. 15,00,000, and "2 crore" resolves to Rs. 2,00,00,000) — with or without an explicit currency marker. Age is pulled separately via an "NN years old" pattern and is stripped out of the query *before* amount extraction runs, so a stated age (e.g. "I am 30 years old") is never mistaken for the income figure. If the query mentions a foreign currency code (USD, EUR, GBP, etc.) with no INR amount given, the calculator asks for the INR-converted figure instead of guessing.
---
6. Prompt Building (non-calculation questions only)
```python
prompt = build_prompt(context, query)
```
Assembles a strict, context-only instruction prompt: answer only from the retrieved text, use Indian number formatting, don't mix old-regime and new-regime rules, and say so explicitly if the answer isn't in the context. If FAISS returns no chunks at all for a query, the app skips the LLM call and returns a direct "couldn't find relevant information" message instead.
---
7. LLM Call
```python
response = client.chat.completions.create(
    model=model_name,      # default: openai/gpt-oss-120b
    temperature=0.1,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
```
Sent via the standard `openai` SDK, pointed at Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`).
---
8. Response Assembly
Both branches (calculator and LLM) return the same shape: `{"answer": ..., "sources": [...]}`, where `sources` are the first 250 characters of each retrieved chunk — shown in the UI's "View Source Context" expander.
---
🔧 Troubleshooting
❌ `ModuleNotFoundError: No module named 'langchain'`
```bash
pip install -r requirements.txt
```
---
❌ `ValueError: GROQ_API_KEY not found`
Make sure you created a `.env` file in the project root containing:
```env
GROQ_API_KEY=your-groq-api-key
```
---
❌ `FileNotFoundError: No knowledge base file found`
Make sure `tax_data.txt` (or `tax_data_.txt`) is in the same folder as `app.py`.
---
❌ `ImportError: The 'faiss' package is not installed`
The FAISS retriever depends on `faiss-cpu`. If installation failed or was skipped, install it explicitly:
```bash
pip install faiss-cpu
```
---
❌ Embedding model fails to load / first run seems stuck downloading
`sentence-transformers` needs an internet connection the first time it runs, to download `all-MiniLM-L6-v2` (~80 MB) from Hugging Face. If it's hanging or failing, check your connection and try again — after the first successful run, the model is cached locally and no further downloads happen.
---
❌ Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```
---
🎓 Project Architecture Diagram
```text
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (app.py)                  │
│                     [Streamlit Web App]                     │
└────────────────────────┬──────────────────────────────────-─┘
                          │ User Query
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG CORE (rag_core.py)                     │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Text Loader │──►│Text Splitter │──►│ Embed + FAISS    │ │
│  │ tax_data.txt │   │ (600 / 100)  │   │ IndexFlatIP (k=6)│ │
│  └──────────────┘   └──────────────┘   └────────┬─────────┘ │
│                                                  │           │
│                          ┌───────────────────────┴──────┐   │
│                          ▼                               ▼   │
│              Tax-calculation question?          General question │
│                          │                               │   │
│                          ▼                               ▼   │
│           Python slab calculator            Retrieve top-6 chunks │
│      (old/new regime, cess, 87A rebate)      → build prompt      │
│                          │                    → call Groq LLM     │
│                          └───────────────┬───────────────┘   │
│                                          ▼                    │
│                            Answer + Source Chunks            │
└─────────────────────────────────────────────────────────────┘
```
---
🚧 Limitations
The knowledge base is a static text file — it does not automatically reflect future Budget changes unless `tax_data.txt` is manually updated.
The deterministic calculator computes tax based on slabs, 4% cess, and the Section 87A rebate only; it does not account for surcharge or additional deductions/exemptions unless explicitly stated in the query.
The app does not maintain conversation history — each question is answered independently, with no memory of earlier turns.
Retrieval quality depends on a general-purpose embedding model (`all-MiniLM-L6-v2`); it is not fine-tuned specifically on Indian tax terminology.
This is an educational/demonstration project, not a substitute for professional tax advice.
---
🔮 Future Enhancements
Re-ranking retrieved chunks (e.g. with a cross-encoder) for higher-precision context on ambiguous questions.
A pipeline to refresh `tax_data.txt` automatically when tax rules change each Budget.
Multi-turn conversational memory, so follow-up questions can reference earlier context.
Basic authentication and query logging for multi-user deployments.
Surcharge and additional-deduction support in the deterministic calculator.
---
📝 License
This project is created for educational purposes as part of a Data Science capstone project. It is not a substitute for professional tax advice — consult a qualified CA for real tax decisions.
---
Built with LangChain + Sentence-Transformers + FAISS + Groq + Streamlit

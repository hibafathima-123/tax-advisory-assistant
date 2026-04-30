# AI-Powered Tax Advisory Assistant

This project is a Streamlit-based RAG app for Indian income tax Q&A.

It now uses:
- Groq for answer generation
- A local TF-IDF retriever for document search
- `tax_data.txt` as the knowledge base

This setup avoids the Windows `torch` / `fbgemm.dll` issue that was breaking the old Hugging Face path.

## Project Files

- `app.py` - Streamlit frontend
- `rag_core.py` - retrieval + LLM pipeline
- `tax_data.txt` - knowledge base
- `requirements.txt` - Python dependencies
- `.env` - local environment variables
- `.env.example` - template environment file

## Setup

### 1. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Add your Groq API key

Open `.env` and update:

```env
GROQ_API_KEY=your-real-groq-api-key
GROQ_MODEL=llama-3.1-8b-instant
```

You can get a Groq API key from:
- https://console.groq.com/

## Run the App

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then open:
- `http://localhost:8501`

## Why the Old Error Happened

Your screenshot showed:

`[WinError 126] ... Error loading ... torch\lib\fbgemm.dll`

That came from the previous local Hugging Face embedding/model path, which depended on `torch` on Windows. The app has been changed so it no longer uses that runtime path.

## Notes

- The app will fail at startup if `GROQ_API_KEY` is missing or still set to the placeholder value.
- Retrieval is still local, so your `tax_data.txt` content remains the source of truth.
- Groq is used only for answer generation after relevant text chunks are retrieved.

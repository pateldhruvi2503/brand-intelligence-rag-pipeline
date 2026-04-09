# Brand Intelligence RAG Pipeline

> A hybrid RAG system that transforms 17 heterogeneous brand data sources into an intelligent Q&A interface — enabling stakeholders to ask brand performance questions in plain English and get instant answers instead of manually searching across files.

**Powered by RAG + Pandas + Gemini**

---

## What This Project Does

Traditional brand analytics requires manually digging through multiple data files to answer a single question. This system replaces that workflow with a conversational AI interface.

**Before:** Open various files, search manually, piece together an answer
**After:** Type your question, get an instant answer with source citations

---

## Architecture

### 1. 9-Agent Data Cleaning Pipeline
Engineered a parallel data cleaning workflow using multi-agent terminal sessions — each agent owned one data source end to end, running simultaneously to automate cleaning across various files:

| Agent | Data Source | What It Does |
|---|---|---|
| Agent-Social | Social mention volume + sentiment | Audits completeness, calculates net sentiment, standardizes to monthly |
| Agent-Search | Branded + comparative + bottom-funnel keywords | Aggregates keyword groups, cleans date formats |
| Agent-GoogleTrends | Google Trends relative interest index | Pulls as single 5-year query, standardizes 0–100 scale |
| Agent-Web | Web traffic overview + by section + conversion events | Aggregates daily to monthly, flags structural breaks |
| Agent-Sales | Sell-through units + market share | Creates 1-month lag on sell-through for Purchase Intent proxy |
| Agent-AdSpend | Ad spend monthly + campaign calendar | Standardizes currency units, aligns campaign dates |
| Agent-Tracker | Brand tracker survey scores | Builds wave metadata with fielding dates and sample sizes |
| Agent-Events | Event registry | Tags product launches, competitor events by construct affected |
| Agent-Supporting | COVID dummy + macro overlays | Creates COVID binary flag, sources Consumer Confidence Index |

### 2. Hybrid RAG System

Pure RAG struggles with time-series data, so this system uses intelligent routing:

- **RAG (30%)** — insight/why questions
- **Pandas (40%)** — trend/time-series questions
- **Both (30%)** — complex questions

**RAG Layer:** HuggingFace embeddings → FAISS vector store → MMR retrieval → Gemini 2.5 Flash
**Router:** Gemini classifies each question and picks the right engine automatically

### 3. Validation
Pearson correlation across 13 tracker wave points:
- Branded Search vs Awareness: r = 0.83
- Net Sentiment vs Consideration: r = 0.88
- Sell-Through vs Purchase Intent: r = 0.91

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Data Processing | Python, Pandas |
| Frontend | Streamlit |
| Synthetic Data | NumPy, Cholesky decomposition |

---

## How to Run

1. Clone the repo
git clone https://github.com/pateldhruvi2503/brand-intelligence-rag-pipeline.git

2. Install dependencies
pip install -r requirements.txt

3. Add your Gemini API key — create a .env file
GEMINI_API_KEY=your_key_here

4. Generate dummy data
python generate_dummy_data.py

5. Run the app
streamlit run streamlit_app.py

---

## Sample Questions

- "What campaigns drove the highest purchase intent?"
- "What happened during COVID?"
- "Show branded search trends in 2023"
- "Which signals correlate most with awareness?"
- "Compare all three constructs in 2023"

---


import os
import glob
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Config — all tunable settings in one place ─────────────────────
@dataclass
class RAGConfig:
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval settings
    fetch_k: int = 100       # candidates FAISS considers
    top_k: int = 30          # chunks sent to Gemini

    # LLM settings
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0  # 0 = deterministic, factual answers

    # Paths
    data_path: str = "data/processed/*.csv"
    vectorstore_path: str = "brand_vectorstore"

config = RAGConfig()

# ── 2. Load API key ───────────────────────────────────────────────────
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# ── 3. Load Gemini ────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=config.model,
    google_api_key=gemini_key,
    temperature=config.temperature
)

# ── 4. Load all CSVs into pandas ──────────────────────────────────────
print("Loading data files...")
dataframes = {}
csv_files = glob.glob(config.data_path)

for filepath in csv_files:
    name = os.path.basename(filepath).replace(".csv", "")
    dataframes[name] = pd.read_csv(filepath)

print(f"Loaded {len(dataframes)} files into pandas")

# ── 5. Load/build FAISS vector store ──────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name=config.embedding_model
)

if os.path.exists(config.vectorstore_path):
    print("Loading existing vector store — skipping re-embedding!")
    vectorstore = FAISS.load_local(
        config.vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building vector store — takes 1-2 mins first time...")
    documents = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        for _, row in df.iterrows():
            row_text = f"Source: {filename}\n"
            for col, val in row.items():
                row_text += f"{col}: {val}\n"
            documents.append(Document(
                page_content=row_text,
                metadata={"source": filename}
            ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(config.vectorstore_path)
    print("Vector store saved!")

# ── 6. Build RAG chain ────────────────────────────────────────────────
prompt_template = """
You are an expert brand analyst. Use ONLY the data provided.
Always cite which source file you are drawing from.
Where relevant, mention correlations between signals.
If data is insufficient, say so clearly — do not guess.

Context from brand data:
{context}

Question: {question}

Answer with:
1. Direct answer with specific numbers
2. Which files the data came from
3. Any signal relationships or correlations relevant to the answer
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": config.top_k,
        "fetch_k": config.fetch_k
    }
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ── 7. Gemini Router ──────────────────────────────────────────────────
def classify_question(question):
    routing_prompt = f"""
You are a question classifier for a brand analytics chatbot.

Classify this question into ONE category:

PANDAS — use when question needs:
- trends over time, full date ranges
- COVID period analysis
- exact number lookups across all months
- comparing multiple years of data
- aggregations (sum, average, max, min)

RAG — use when question needs:
- insight, reasoning, correlations
- campaign analysis, what drove something
- relationships between signals
- why something happened

BOTH — use when question needs data AND insight together

Question: {question}
Reply with ONLY one word: PANDAS, RAG, or BOTH
"""
    response = llm.invoke(routing_prompt)
    route = response.content.strip().upper()
    if "PANDAS" in route:
        return "PANDAS"
    elif "BOTH" in route:
        return "BOTH"
    else:
        return "RAG"

# ── 8. Pandas engine ──────────────────────────────────────────────────
def pandas_answer(question):
    data_summary = ""
    for name, df in dataframes.items():
        data_summary += f"\n--- {name} ---\n"
        data_summary += f"Columns: {list(df.columns)}\n"
        data_summary += df.to_string(max_rows=200) + "\n"

    pandas_prompt = f"""
You are an expert brand analyst with access to the full dataset.
Answer using the data below. Be specific with numbers and dates.

Full Dataset:
{data_summary}

Question: {question}

Answer with:
1. Direct answer with specific numbers
2. A markdown table showing the relevant data
3. Any patterns or insights you notice
"""
    response = llm.invoke(pandas_prompt)
    return response.content

# ── 9. Main chat loop ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Brand Intelligence Chatbot v2 — Hybrid RAG + Pandas")
print("="*60)
print("Auto-detects question type and routes to best engine")
print(f"Config: model={config.model} | temp={config.temperature} | top_k={config.top_k}")
print("Type 'quit' to exit\n")

while True:
    question = input("Ask anything about brand data:\n> ")

    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    print("\nClassifying question...")
    route = classify_question(question)
    print(f"Routing to: [{route}]")
    print("Searching brand data...\n")

    if route == "PANDAS":
        answer = pandas_answer(question)
        print(f"Answer:\n{answer}")
        print(f"\n[Answered by: Pandas engine]")

    elif route == "RAG":
        result = qa_chain.invoke({"query": question})
        print(f"Answer:\n{result['result']}")
        sources = set(
            doc.metadata["source"]
            for doc in result["source_documents"]
        )
        print(f"\nSources used: {', '.join(sources)}")
        print(f"\n[Answered by: RAG engine]")

    elif route == "BOTH":
        pandas_result = pandas_answer(question)
        rag_result = qa_chain.invoke({"query": question})

        combine_prompt = f"""
You are a brand analyst. Combine these two analyses into one clear answer.

Data Analysis:
{pandas_result}

Insight Analysis:
{rag_result['result']}

Write a unified answer with:
1. Key insight
2. Supporting data table
3. Signal relationships and correlations
"""
        final = llm.invoke(combine_prompt)
        print(f"Answer:\n{final.content}")
        sources = set(
            doc.metadata["source"]
            for doc in rag_result["source_documents"]
        )
        print(f"\nSources used: {', '.join(sources)}")
        print(f"\n[Answered by: Pandas + RAG combined]")

    print("\n" + "-"*60 + "\n")

import os
import glob
import pandas as pd

import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
import plotly.express as px

from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    fetch_k: int = 100
    top_k: int = 30
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    data_path: str = "data/processed/*.csv"
    vectorstore_path: str = "brand_vectorstore"

config = RAGConfig()

# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brand Intelligence",
    page_icon="🔵",
    layout="wide"
)

# ── Load API key ──────────────────────────────────────────────────────
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Brand Intelligence")
    st.markdown("*Powered by RAG + Pandas + Gemini*")
    st.divider()

    # Key metrics from brand tracker
    st.markdown("### Latest Brand Scores")
    try:
        bt = pd.read_csv("data/processed/brand_tracker_cleaned.csv")
        latest_wave = bt["wave_id"].iloc[-1]
        awareness = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Awareness")]["score"].values[0]
        consideration = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Consideration")]["score"].values[0]
        pi = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Purchase Intent")]["score"].values[0]
        st.metric("Awareness", f"{awareness}", delta="Latest wave")
        st.metric("Consideration", f"{consideration}")
        st.metric("Purchase Intent", f"{pi}")
        st.caption(f"Wave: {latest_wave}")
    except:
        st.info("Load data to see metrics")

    st.divider()
    st.markdown("### Sample Questions")
    sample_questions = [
        "What campaigns drove highest purchase intent?",
        "What happened during COVID?",
        "Show branded search trends in 2023",
        "Which signals correlate with awareness?",
        "Compare all three constructs in 2023"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.sample_q = q

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Load data (cached so it only runs once) ───────────────────────────
@st.cache_resource
def load_everything():
    llm = ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=gemini_key,
        temperature=config.temperature
    )

    # Load pandas dataframes
    dataframes = {}
    csv_files = glob.glob(config.data_path)
    for filepath in csv_files:
        name = os.path.basename(filepath).replace(".csv", "")
        dataframes[name] = pd.read_csv(filepath)

    # Load vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )

    if os.path.exists(config.vectorstore_path):
        vectorstore = FAISS.load_local(
            config.vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
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

    # Build RAG chain
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
3. Any signal relationships or correlations
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.top_k, "fetch_k": config.fetch_k}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return llm, dataframes, qa_chain

# ── Helper functions ──────────────────────────────────────────────────
def classify_question(llm, question):
    routing_prompt = f"""
Classify this question into ONE category:
PANDAS — trends over time, full date ranges, COVID analysis, aggregations
RAG — insights, correlations, campaign analysis, why something happened
BOTH — needs data AND insight together

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

def pandas_answer(llm, dataframes, question):
    data_summary = ""
    for name, df in dataframes.items():
        data_summary += f"\n--- {name} ---\n"
        data_summary += f"Columns: {list(df.columns)}\n"
        data_summary += df.to_string(max_rows=200) + "\n"

    pandas_prompt = f"""
You are an expert brand analyst. Answer using the data below.
Be specific with numbers and dates.
Always include a markdown table of the relevant data.

Full Dataset:
{data_summary}

Question: {question}

Answer with:
1. Direct answer with specific numbers
2. A markdown table of relevant data
3. Key patterns or insights
"""
    response = llm.invoke(pandas_prompt)
    return response.content

def get_answer(llm, dataframes, qa_chain, question):
    route = classify_question(llm, question)
    sources = []

    if route == "PANDAS":
        answer = pandas_answer(llm, dataframes, question)
        return answer, sources

    elif route == "RAG":
        result = qa_chain.invoke({"query": question})
        sources = list(set(
            doc.metadata["source"]
            for doc in result["source_documents"]
        ))
        return result["result"], sources

    else:  # BOTH
        pandas_result = pandas_answer(llm, dataframes, question)
        rag_result = qa_chain.invoke({"query": question})
        sources = list(set(
            doc.metadata["source"]
            for doc in rag_result["source_documents"]
        ))
        combine_prompt = f"""
You are a brand analyst. Combine these analyses into one clear answer.

Data Analysis:
{pandas_result}

Insight Analysis:
{rag_result['result']}

Write a unified answer with:
1. Key insight in plain English
2. Supporting data table
3. Signal relationships and correlations
"""
        final = llm.invoke(combine_prompt)
        return final.content, sources

# ── Main app ──────────────────────────────────────────────────────────
st.title("Brand Intelligence Chatbot")
st.caption("Ask anything about brand data — awareness, consideration, purchase intent, campaigns, signals")

# Load everything
with st.spinner("Loading data and AI models..."):
    llm, dataframes, qa_chain = load_everything()

# Quick metrics row at top
st.divider()
col1, col2, col3, col4 = st.columns(4)
try:
    bt = pd.read_csv("data/processed/brand_tracker_cleaned.csv")
    bs = pd.read_csv("data/processed/branded_search_cleaned.csv")
    latest_wave = bt["wave_id"].iloc[-1]
    awareness = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Awareness")]["score"].values[0]
    consideration = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Consideration")]["score"].values[0]
    pi = bt[(bt["wave_id"]==latest_wave) & (bt["construct"]=="Purchase Intent")]["score"].values[0]
    latest_search = bs["branded_search_volume"].iloc[-1]

    with col1:
        st.metric("Awareness", f"{awareness}")
    with col2:
        st.metric("Consideration", f"{consideration}")
    with col3:
        st.metric("Purchase Intent", f"{pi}")
    with col4:
        st.metric("Latest Branded Search", f"{latest_search:,.0f}")
except:
    pass

st.divider()

# Brand trends chart
with st.expander("Brand Health Trends", expanded=True):
    try:
        bt = pd.read_csv("data/processed/brand_tracker_cleaned.csv")
        fig = px.line(
            bt,
            x="wave_id",
            y="score",
            color="construct",
            markers=True,
            title="Brand Tracker — All Constructs Over Time",
            labels={"wave_id": "Wave", "score": "Score", "construct": "Construct"},
            color_discrete_map={
                "Awareness": "#0068b5",
                "Consideration": "#00aeef",
                "Purchase Intent": "#00c7fd"
            }
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Chart will appear once data loads")

st.divider()

# ── Chat interface ────────────────────────────────────────────────────
st.markdown("### Ask the Brand Analyst")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for s in message["sources"]:
                    st.caption(f"📄 {s}")

# Handle sample question clicks from sidebar
if "sample_q" in st.session_state:
    prompt = st.session_state.sample_q
    del st.session_state.sample_q
else:
    prompt = st.chat_input("Ask about brand data...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Analysing data..."):
            answer, sources = get_answer(llm, dataframes, qa_chain, prompt)
        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for s in sources:
                    st.caption(f"📄 {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
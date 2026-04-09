import os
import glob
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load API key from .env ─────────────────────────────────────────
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# ── 2. Load CSVs ──────────────────────────────────────────────────────
print("Loading data files...")

csv_files = glob.glob("data/processed/*.csv")
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

print(f"Loaded {len(documents)} rows across {len(csv_files)} files")

# ── 3. Split into chunks ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ── 4. Embed with HuggingFace ─────────────────────────────────────────
print("Embedding chunks — first time takes 1-2 mins, downloads ~90MB...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── 5. Store in FAISS ─────────────────────────────────────────────────
if os.path.exists("brand_vectorstore"):
    print("Loading existing vector store — skipping re-embedding!")
    vectorstore = FAISS.load_local(
        "brand_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building vector store for first time...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("brand_vectorstore")
    print("Vector store saved!")

# ── 6. Set up Gemini ──────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.1
)

# ── 7. Domain-specific prompt ─────────────────────────────────────────
prompt_template = """
You are an expert brand analyst. You have access to brand tracking data
including Awareness, Consideration, and Purchase Intent metrics, along with
proxy signals like branded search volume, social sentiment, sell-through data,
and Google Trends.

Use ONLY the data provided below to answer the question.
If the data doesn't contain enough information, say so clearly.
Always cite which data source (filename) you're drawing from.
Where relevant, mention correlations between signals.

Context from brand data:
{context}

Question: {question}

Answer (be specific, cite sources, mention signal relationships):
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ── 8. Build the RAG chain ────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 30, "fetch_k": 100}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ── 9. Chat loop ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("Brand RAG Chatbot — powered by your cleaned data")
print("="*60)
print("Type 'quit' to exit\n")

while True:
    question = input("Ask a question about brand data:\n> ")

    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    print("\nSearching brand data...")
    result = qa_chain.invoke({"query": question})

    print(f"\nAnswer:\n{result['result']}")

    sources = set(doc.metadata["source"] for doc in result["source_documents"])
    print(f"\nSources used: {', '.join(sources)}")
    print("\n" + "-"*60 + "\n")
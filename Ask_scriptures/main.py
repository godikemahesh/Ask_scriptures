
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime

# ----------------------------- App Configuration -----------------------------
st.set_page_config(
    page_title="Ask Scriptures AI - Gita Assistant",
    page_icon="🕊️",
    layout="wide"
)

# ----------------------------- UI Elements -----------------------------
st.markdown("""
    <style>
    .title {
        font-size:48px;
        font-weight: bold;
        color: #1e3c72;
        text-align: center;
    }
    .subtitle {
        font-size:24px;
        text-align: center;
        color: #555;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: gray;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Ask Scriptures AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Spiritual Guide from the Bhagavad Gita 🕊️</div>', unsafe_allow_html=True)

st.markdown("---")

# ----------------------------- Load Resources -----------------------------
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("gita_faiss.index")

@st.cache_resource
def load_chunks():
    with open("gita_chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

index = load_faiss_index()
chunks = load_chunks()
model = load_model()

client = OpenAI(
    api_key="mdb_0JXfDNMjdfKy4BB3mxQWQwr9Ol2jvwSIYfduoeMAiFa1",
    base_url='https://llm.mdb.ai/'
)

# ----------------------------- Core Logic -----------------------------
def get_gita_answer(question):
    query_vector = model.encode([question])
    D, I = index.search(np.array(query_vector), k=4)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
You are an AI spiritual assistant trained on Bhagavad Gita.
Based on the following Gita verses, answer the question with meaning
from given gita Context only.do not hallucinate
contect:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content.strip()

# ----------------------------- Chat Interface -----------------------------
st.subheader("🔍 Ask a Question from the Gita")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Type your question here...")

if question:
    st.session_state.chat_history.append(("You", question))
    answer = get_gita_answer(question)
    st.session_state.chat_history.append(("Gita AI", answer))

# ----------------------------- Display Chat -----------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ----------------------------- Footer -----------------------------
st.markdown("""
    <div class="footer">
        🌟 Powered by SURAJ AI | Developed by Mahi 🚀
    </div>
""", unsafe_allow_html=True)

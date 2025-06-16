import base64

import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime

# ----------------------------- App Configuration -----------------------------
st.set_page_config(
    page_title="Ask Gita AI - Spiritual Chat Assistant",
    page_icon="🕊️",
    layout="wide"
)

# ----------------------------- UI Elements -----------------------------
st.markdown("""
    <style>
    .title {
        font-size:52px;
        font-weight: bold;
        color: #1e3c72;
        text-align: center;
        letter-spacing: 1px;
    }
    .subtitle {
        font-size:22px;
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
    .gita-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">Ask Scripures AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Spiritual guide from Bhagavad Gita 🕊️</div>', unsafe_allow_html=True)

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

# ----------------------------- Greeting Logic -----------------------------
greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "namaste"]
thanks_keywords = ["thank", "thanks", "great", "awesome", "good job", "nice"]
sample_questions = [
    "How to control the mind?",
    "What is the path to peace according to the Gita?",
    "How to deal with fear and anxiety?",
    "What is Karma Yoga?"
]

# ----------------------------- Chat Interface -----------------------------
st.subheader("🔍 Ask a Question from the Gita")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Type your question here...")

if question:
    user_input = question.lower()

    if any(greet in user_input for greet in greeting_keywords):
        reply = "Namaste 🙏 How can I assist you today with the wisdom of the Gita?"
        suggestion_text = "Here are a few things you can ask:\n" + "\n".join([f"- {q}" for q in sample_questions])
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Gita AI", reply + "\n\n" + suggestion_text))

    elif any(word in user_input for word in thanks_keywords):
        reply = "You're most welcome 🙏 May your path be full of clarity and peace."
        suggestion_text = "Would you like to explore more? Try asking something like:\n" + "\n".join([f"- {q}" for q in sample_questions])
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Gita AI", reply + "\n\n" + suggestion_text))

    else:
        st.session_state.chat_history.append(("You", question))
        answer = get_gita_answer(question)
        st.session_state.chat_history.append(("Gita AI", answer))

# ----------------------------- Display Chat -----------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "Gita AI":
            # Load local image and encode it as base64
            with open("gita_dp.jpg", "rb") as img_file:
                img_bytes = img_file.read()
                b64_img = base64.b64encode(img_bytes).decode()
                img_tag = f"<img src='data:image/png;base64,{b64_img}' class='gita-avatar'>"
            st.markdown(
                f"{img_tag}<span>{msg}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(msg)

# ----------------------------- Footer -----------------------------
st.markdown("""
    <div class="footer">
        🌟 Powered by SURAJ AI | Developed by Mahi 🚀
    </div>
""", unsafe_allow_html=True)

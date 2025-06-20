import base64
from google.oauth2.service_account import Credentials
import gspread
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime
from groq import Groq

# ----------------------------- Google Sheet Setup -----------------------------
def append_chat_to_sheet(user_input, gita_response):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1NDpRh9mBoTy3tffAegGLMBRxcdPpQcWNpLVIcNtBCSc").sheet1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, user_input, gita_response])

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Ask Gita AI - Spiritual Chat Assistant",
    page_icon="🕊️",
    layout="wide"
)

# ----------------------------- Styling -----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #FFE3C5, #FFD6EC, #C2F0FC);
    }
    .block-container {
        padding-top: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 56px;
        font-weight: bold;
        background: -webkit-linear-gradient(#ff6b6b, #fbc531);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 5px #f39c12, 0 0 10px #f39c12; }
        100% { text-shadow: 0 0 20px #f39c12, 0 0 30px #e67e22; }
    }
    .subtitle {
        font-size: 22px;
        color: #00FFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f9f9f9;
        font-size: 14px;
        border-top: 1px solid #ccc;
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

# ----------------------------- App Header -----------------------------
st.markdown('<div class="title">✨ Ask Scriptures AI ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Spiritual Guide from the Bhagavad Gita 🕉️🌸</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------- Load Resources -----------------------------
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("Ask_scriptures/gita_faiss.index")

@st.cache_resource
def load_chunks():
    with open("Ask_scriptures/gita_chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

index = load_faiss_index()
chunks = load_chunks()
model = load_model()

client = Groq(api_key=st.secrets["gcp_service_account"]["groq_api"])

# ----------------------------- Gita QA Logic -----------------------------
def get_gita_answer(question):
    query_vector = model.encode([question])
    D, I = index.search(np.array(query_vector), k=4)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
You are an AI spiritual assistant trained on Bhagavad Gita.
Based on the following Gita verses, answer the question with meaning
from given gita Context only. Do not hallucinate.

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content.strip()

# ----------------------------- Chat Interaction -----------------------------
greeting_keywords = ["hello", "hi","hii", "hey", "good morning", "good evening", "namaste"]
thanks_keywords = ["thank", "thanks", "great", "awesome", "good","good job", "nice","super"]
sample_questions = [
    "How to control the mind?",
    "What is the path to peace according to the Gita?",
    "How to deal with fear and anxiety?",
    "What is Karma Yoga?"
]

st.markdown("""
<h3 style='
    color:#00FFFF;
    text-shadow: 0 0 10px #00FFFF;
    font-weight:bold;
'>🔍 Ask a Question from the Gita</h3>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Type your question here...")

if question:
    user_input = question.lower()

    if any(greet == user_input for greet in greeting_keywords):
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
        append_chat_to_sheet(question, answer)

# ----------------------------- Display Chat -----------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "Gita AI":
            with open("Ask_scriptures/gita_dp.jpg", "rb") as img_file:
                img_bytes = img_file.read()
                b64_img = base64.b64encode(img_bytes).decode()
                img_tag = f"<img src='data:image/png;base64,{b64_img}' class='gita-avatar'>"
            st.markdown(
    f"""
    <div style='background-color:#fff7e6;padding:15px;border-radius:10px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
        {img_tag}<span style='font-size:16px; color:#2c3e50; font-family:"Segoe UI", sans-serif;'>{msg}</span>
    </div>
    """,
    unsafe_allow_html=True
)

        else:
            st.markdown(
                f"""
                <div style='background-color:#fff7e6;padding:15px;border-radius:10px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
                    <span style='font-size:16px;font-weight:bold;'>{msg}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------- Footer -----------------------------
st.markdown("""
    <div class="footer">
        🌟 <span style="color:#d35400;font-weight:bold;">Powered by SURAJ AI</span> | Developed by <span style="color:#8e44ad;font-weight:bold;">Mahi 🚀</span><br>
        <span style="font-size:12px;color:#aaa;">🕊️ Embrace the teachings of Gita, live with purpose.</span>
    </div>
""", unsafe_allow_html=True)

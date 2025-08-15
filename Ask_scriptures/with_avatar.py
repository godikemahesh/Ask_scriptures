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
def append_chat_to_sheet(user_input, gita_response, context):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1NDpRh9mBoTy3tffAegGLMBRxcdPpQcWNpLVIcNtBCSc").sheet1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, user_input, gita_response, context])

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Ask Scriptures AI - Spiritual Guidance",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------- Professional Styling -----------------------------
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main container styling */
    .main {
        background: linear-gradient(180deg, #FAFBFC 0%, #F5F7FA 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
        margin: 0 auto;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #E5E7EB, transparent);
        margin: 2rem 0;
    }
    
    /* Chat section header */
    .chat-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Message styling */
    .user-message {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border: 1px solid #E5E7EB;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .message-content {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #374151;
    }
    
    .message-role {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B7280;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Avatar styling */
    .avatar {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
    }
    
    .footer-credits {
        margin-top: 0.5rem;
        color: #9CA3AF;
        font-size: 0.8rem;
    }
    
    /* Chat input styling */
    .stChatInput {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
    }
    
    /* Button styling */
    .stButton button {
        background: #4F46E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background 0.2s;
    }
    
    .stButton button:hover {
        background: #4338CA;
    }
    
    /* Remove default Streamlit styling */
    .css-1d391kg, .css-1d391kg p {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    
    /* Sample questions styling */
    .sample-questions {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #6B7280;
    }
    
    .sample-questions ul {
        margin: 0.5rem 0 0 1rem;
        padding: 0;
    }
    
    .sample-questions li {
        margin: 0.25rem 0;
        color: #4B5563;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- App Header -----------------------------
st.markdown("""
    <div class="app-header">
        <div class="main-title">🕉️ Ask Scriptures AI</div>
        <div class="subtitle">Spiritual guidance from the Bhagavad Gita</div>
    </div>
    <div class="divider"></div>
""", unsafe_allow_html=True)

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
    return response.choices[0].message.content.strip(), context

# ----------------------------- Chat Interaction -----------------------------
greeting_keywords = ["hello", "hi", "hii", "hey", "good morning", "good evening", "namaste"]
thanks_keywords = ["thank", "thanks", "great", "awesome", "good", "good job", "nice", "super"]
sample_questions = [
    "How to control the mind?",
    "What is the path to peace according to the Gita?",
    "How to deal with fear and anxiety?",
    "What is Karma Yoga?"
]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
st.markdown("""
    <div class="chat-header">
        <span>💬</span>
        <span>Ask your question</span>
    </div>
""", unsafe_allow_html=True)

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"""
            <div class="user-message">
                <div class="message-role">
                    <span>👤</span>
                    <span>You</span>
                </div>
                <div class="message-content">{msg}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Format sample questions if present
        if "Here are a few things you can ask:" in msg or "Would you like to explore more?" in msg:
            parts = msg.split("\n")
            main_message = parts[0]
            if len(parts) > 1:
                questions_html = "<div class='sample-questions'><p>Suggested questions:</p><ul>"
                for part in parts[1:]:
                    if part.strip().startswith("-"):
                        questions_html += f"<li>{part[1:].strip()}</li>"
                questions_html += "</ul></div>"
                formatted_msg = main_message + questions_html
            else:
                formatted_msg = main_message
        else:
            formatted_msg = msg.replace("\n", "<br>")
        
        st.markdown(f"""
            <div class="ai-message">
                <div class="message-role">
                    <span>🕉️</span>
                    <span>Gita AI</span>
                </div>
                <div class="message-content">{formatted_msg}</div>
            </div>
        """, unsafe_allow_html=True)

# Chat input
question = st.chat_input("Type your spiritual question here...")

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
        with st.spinner("Searching the Gita for wisdom..."):
            answer, context = get_gita_answer(question)
        st.session_state.chat_history.append(("Gita AI", answer))
        append_chat_to_sheet(question, answer, context)
    
    st.rerun()

# ----------------------------- Footer -----------------------------
st.markdown("""
    <div class="footer">
        <div>Powered by SURAJ AI | Spiritual guidance from ancient wisdom</div>
        <div class="footer-credits">Developed</div>
    </div>
""", unsafe_allow_html=True)

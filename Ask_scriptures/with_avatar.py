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
    page_title="Ask Scriptures AI",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------- Grok-Style Dark Theme -----------------------------
st.markdown("""
    <style>
    /* Import Inter font like Grok */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme background */
    .stApp {
        background-color: #000000;
    }
    
    .main {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        padding: 0;
    }
    
    .block-container {
        max-width: 800px;
        padding: 1rem 1rem 6rem 1rem;
        margin: 0 auto;
        background-color: #000000;
    }
    
    /* Header section exactly like Grok */
    .grok-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }
    
    .grok-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .logo-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    
    /* Chat messages exactly like Grok */
    .chat-container {
        margin-top: 2rem;
    }
    
    .message-wrapper {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        padding: 0;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 6px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        margin-top: 2px;
    }
    
    .user-avatar {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .message-content {
        flex: 1;
        color: #e4e4e4;
        font-size: 15px;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .message-content p {
        margin: 0;
        white-space: pre-wrap;
    }
    
    /* Code blocks like Grok */
    .message-content code {
        background: #1a1a1a;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 14px;
        color: #e4e4e4;
        font-family: 'SF Mono', Monaco, monospace;
    }
    
    .message-content pre {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    /* Suggested questions box */
    .suggestions-box {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .suggestions-title {
        color: #888;
        font-size: 13px;
        margin-bottom: 0.75rem;
        font-weight: 500;
    }
    
    .suggestion-item {
        color: #e4e4e4;
        font-size: 14px;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1a1a1a;
        cursor: pointer;
        transition: color 0.2s;
    }
    
    .suggestion-item:last-child {
        border-bottom: none;
    }
    
    .suggestion-item:hover {
        color: #667eea;
    }
    
    /* Input area exactly like Grok */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #000000;
        border-top: 1px solid #2a2a2a;
        padding: 1rem 0;
        z-index: 999;
    }
    
    .stChatInput > div {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .stChatInput textarea {
        background: #0a0a0a !important;
        border: 1px solid #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        resize: none !important;
        min-height: 48px !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #666 !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea !important;
        color: #667eea !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling like Grok */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2a2a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3a3a3a;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 4rem 2rem;
        color: #888;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Quick action buttons like Grok */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        max-width: 500px;
        margin: 2rem auto 0;
    }
    
    .quick-action-btn {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-action-btn:hover {
        border-color: #667eea;
        background: #111;
    }
    
    .quick-action-title {
        color: #e4e4e4;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .quick-action-desc {
        color: #666;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- Header -----------------------------
st.markdown("""
    <div class="grok-header">
        <div class="grok-logo">
            <div class="logo-icon">🕉️</div>
            <div class="logo-text">Ask Scriptures</div>
        </div>
    </div>
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
    {"title": "Control the Mind", "desc": "Techniques from the Gita"},
    {"title": "Path to Peace", "desc": "Finding inner tranquility"},
    {"title": "Dealing with Fear", "desc": "Overcoming anxiety"},
    {"title": "Karma Yoga", "desc": "The path of action"}
]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display welcome message if no chat history
if not st.session_state.chat_history:
    st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">Ask the Bhagavad Gita</div>
            <div class="welcome-subtitle">Discover timeless wisdom and spiritual guidance</div>
            <div class="quick-actions">
    """, unsafe_allow_html=True)
    
    for q in sample_questions:
        st.markdown(f"""
            <div class="quick-action-btn">
                <div class="quick-action-title">{q['title']}</div>
                <div class="quick-action-desc">{q['desc']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"""
                <div class="message-wrapper">
                    <div class="message-avatar user-avatar">👤</div>
                    <div class="message-content">
                        <p>{msg}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Check if message contains suggestions
            if "Here are a few things you can ask:" in msg or "Would you like to explore more?" in msg:
                parts = msg.split("\n")
                main_msg = parts[0]
                
                st.markdown(f"""
                    <div class="message-wrapper">
                        <div class="message-avatar ai-avatar">🕉️</div>
                        <div class="message-content">
                            <p>{main_msg}</p>
                """, unsafe_allow_html=True)
                
                if len(parts) > 2:
                    st.markdown('<div class="suggestions-box"><div class="suggestions-title">Suggested questions</div>', unsafe_allow_html=True)
                    for part in parts[2:]:
                        if part.strip().startswith("-"):
                            st.markdown(f'<div class="suggestion-item">{part[1:].strip()}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="message-wrapper">
                        <div class="message-avatar ai-avatar">🕉️</div>
                        <div class="message-content">
                            <p>{msg}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input (fixed at bottom)
question = st.chat_input("Ask about the Bhagavad Gita...")

if question:
    user_input = question.lower()

    if any(greet == user_input for greet in greeting_keywords):
        reply = "Namaste 🙏 I'm here to help you explore the wisdom of the Bhagavad Gita."
        suggestion_text = "\nHere are a few things you can ask:\n" + "\n".join([f"- {q['title']}: {q['desc']}" for q in sample_questions])
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Gita AI", reply + suggestion_text))

    elif any(word in user_input for word in thanks_keywords):
        reply = "You're welcome! May the teachings bring you peace and clarity. 🙏"
        suggestion_text = "\nWould you like to explore more? Try asking:\n" + "\n".join([f"- {q['title']}" for q in sample_questions[:2]])
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Gita AI", reply + suggestion_text))

    else:
        st.session_state.chat_history.append(("You", question))
        with st.spinner(""):
            answer, context = get_gita_answer(question)
        st.session_state.chat_history.append(("Gita AI", answer))
        append_chat_to_sheet(question, answer, context)
    
    st.rerun()

# Add some padding at the bottom for the input
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

import base64
from google.oauth2.service_account import Credentials
import gspread
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime, timedelta
from groq import Groq
import hashlib
import re
from collections import defaultdict
import time
import random

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Ask Scriptures AI - Advanced Spiritual Intelligence",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Initialize Session State -----------------------------
def init_session_state():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "spiritual_level": "beginner",
            "interests": [],
            "favorite_verses": [],
            "meditation_streak": 0,
            "total_questions": 0,
            "wisdom_points": 0,
            "last_active": datetime.now(),
            "achievements": [],
            "personality_traits": {"seeker": 70, "devotional": 50, "intellectual": 60},
            "preferred_topics": defaultdict(int),
            "learning_style": "balanced"
        }
    if "memory_bank" not in st.session_state:
        st.session_state.memory_bank = {
            "key_learnings": [],
            "personal_insights": [],
            "recurring_themes": defaultdict(int),
            "emotional_patterns": defaultdict(list),
            "context_threads": {},
            "deep_questions": [],
            "breakthrough_moments": [],
            "wisdom_evolution": []
        }
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
    if "ai_personality" not in st.session_state:
        st.session_state.ai_personality = {
            "empathy_level": 0.8,
            "wisdom_depth": 0.9,
            "personal_connection": 0.7,
            "adaptive_style": True
        }
    if "conversation_flow" not in st.session_state:
        st.session_state.conversation_flow = {
            "current_theme": None,
            "depth_level": 1,
            "emotional_state": "neutral",
            "breakthrough_potential": 0.5
        }
    if "achievement_popup" not in st.session_state:
        st.session_state.achievement_popup = None
    if "thinking_animation" not in st.session_state:
        st.session_state.thinking_animation = False

init_session_state()

# ----------------------------- Google Sheet Setup -----------------------------
def append_chat_to_sheet(session_id, user_input, gita_response, context, sentiment="neutral", topic_tags=[]):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key("1NDpRh9mBoTy3tffAegGLMBRxcdPpQcWNpLVIcNtBCSc").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Enhanced logging with user profile data
        profile_data = json.dumps({
            "wisdom_points": st.session_state.user_profile["wisdom_points"],
            "total_questions": st.session_state.user_profile["total_questions"],
            "personality_traits": st.session_state.user_profile["personality_traits"]
        })
        
        sheet.append_row([
            timestamp, session_id, user_input, gita_response, context, 
            sentiment, ", ".join(topic_tags), profile_data
        ])
    except Exception as e:
        # Enhanced error logging
        st.session_state.memory_bank["key_learnings"].append({
            "timestamp": datetime.now(),
            "type": "system_error",
            "content": f"Sheet logging failed: {str(e)[:100]}"
        })

# ----------------------------- Advanced Grok-Style Theme -----------------------------


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .suggestion-pill::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 107, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .suggestion-pill:hover {
        border-color: #FF6B6B;
        color: #FF6B6B;
        background: rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    .suggestion-pill:hover::before {
        left: 100%;
    }
    
    /* Input Area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(20px);
        border-top: 1px solid #1a1a1a;
        padding: 1.5rem;
        z-index: 1000;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 1rem;
        align-items: flex-end;
    }
    
    .stChatInput {
        flex: 1;
    }
    
    .stChatInput textarea {
        background: rgba(10, 10, 10, 0.9) !important;
        border: 1px solid #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        font-size: 15px !important;
        resize: none !important;
        transition: all 0.3s !important;
        min-height: 56px !important;
        max-height: 120px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.2) !important;
        background: rgba(10, 10, 10, 0.95) !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #666 !important;
        font-style: italic !important;
    }
    
    /* Loading Animation */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.25rem;
        background: rgba(10, 10, 10, 0.8);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        margin: 1rem 0;
        animation: fadeIn 0.3s ease-out;
    }
    
    .thinking-dots {
        display: flex;
        gap: 0.25rem;
    }
    
    .thinking-dot {
        width: 8px;
        height: 8px;
        background: #FF6B6B;
        border-radius: 50%;
        animation: thinking 1.4s ease-in-out infinite;
    }
    
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes thinking {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-12px);
            opacity: 1;
        }
    }
    
    .thinking-text {
        color: #888;
        font-size: 14px;
        font-style: italic;
    }
    
    /* Achievement Toast */
    .achievement-toast {
        position: fixed;
        top: 100px;
        right: 20px;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        padding: 1.25rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        animation: achievementSlide 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        z-index: 2000;
        min-width: 300px;
    }
    
    @keyframes achievementSlide {
        from {
            transform: translateX(400px) scale(0.8);
            opacity: 0;
        }
        to {
            transform: translateX(0) scale(1);
            opacity: 1;
        }
    }
    
    .achievement-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .achievement-icon {
        font-size: 24px;
    }
    
    .achievement-title {
        font-size: 16px;
        font-weight: 600;
    }
    
    .achievement-desc {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Context Thread Visualization */
    .context-thread {
        position: absolute;
        left: 20px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: linear-gradient(to bottom, transparent, rgba(255, 107, 107, 0.3), transparent);
        opacity: 0;
        animation: threadPulse 2s ease-in-out infinite;
    }
    
    @keyframes threadPulse {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    
    /* Memory Visualization */
    .memory-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 24px;
        height: 24px;
        border: 2px solid rgba(78, 205, 196, 0.3);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #4ECDC4;
        animation: memoryPulse 3s ease-in-out infinite;
    }
    
    @keyframes memoryPulse {
        0%, 100% { 
            border-color: rgba(78, 205, 196, 0.3);
            background: transparent;
        }
        50% { 
            border-color: rgba(78, 205, 196, 0.6);
            background: rgba(78, 205, 196, 0.1);
        }
    }
    
    /* Breakthrough Moment Highlight */
    .breakthrough-moment {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 2px solid rgba(255, 215, 0, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .breakthrough-moment::before {
        content: '✨';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 24px;
        animation: sparkle 2s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.7; }
        50% { transform: scale(1.2) rotate(180deg); opacity: 1; }
    }
    
    /* Advanced Animations */
    .ai-typing {
        animation: aiTyping 1.5s ease-in-out infinite;
    }
    
    @keyframes aiTyping {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .wisdom-glow {
        animation: wisdomGlow 3s ease-in-out infinite;
    }
    
    @keyframes wisdomGlow {
        0%, 100% { filter: brightness(1) contrast(1); }
        50% { filter: brightness(1.1) contrast(1.1); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .css-1y4p8pa {
            width: 280px !important;
        }
        
        .welcome-title {
            font-size: 2.5rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .suggestions-container {
            flex-direction: column;
            align-items: center;
        }
        
        .message-wrapper {
            gap: 0.75rem;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            font-size: 16px;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stToolbar {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        border-radius: 4px;
        transition: background 0.3s;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    }
    
    /* Print Styles */
    @media print {
        .sidebar, .input-container, .header-bar {
            display: none !important;
        }
        
        .main {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .message-wrapper {
            break-inside: avoid;
        }
    }
    </style>
""", unsafe_allow_html=True)
# ----------------------------- Load ML Resources -----------------------------
@st.cache_resource
def load_faiss_index():
    try:
        return faiss.read_index("Ask_scriptures/gita_faiss.index")
    except:
        # Fallback: create a simple index
        st.error("FAISS index not found. Please ensure the index file exists.")
        return None

@st.cache_resource
def load_chunks():
    try:
        with open("Ask_scriptures/gita_chunks.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        # Fallback data
        return ["Sample Gita text for demo purposes."]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize resources
index = load_faiss_index()
chunks = load_chunks()
model = load_model()

# Initialize Groq client
try:
    client = Groq(api_key=st.secrets["gcp_service_account"]["groq_api"])
except:
    st.error("Groq API key not found. Please configure your secrets.")
    client = None

# ----------------------------- Advanced Memory System -----------------------------

class AdvancedMemoryManager:
    def __init__(self):
        self.episodic_memory = []
        self.semantic_memory = defaultdict(list)
        self.procedural_memory = {}
        self.emotional_memory = defaultdict(list)
        self.context_windows = {}
        
    def store_interaction(self, user_input, ai_response, context, sentiment, topics, session_id):
        """Store interaction in multiple memory systems"""
        timestamp = datetime.now()
        
        # Episodic Memory - What happened when
        episode = {
            "timestamp": timestamp,
            "user_input": user_input,
            "ai_response": ai_response,
            "context": context,
            "sentiment": sentiment,
            "topics": topics,
            "session_id": session_id,
            "importance_score": self._calculate_importance(user_input, sentiment, topics)
        }
        self.episodic_memory.append(episode)
        
        # Semantic Memory - What we know about topics
        for topic in topics:
            self.semantic_memory[topic].append({
                "content": user_input,
                "timestamp": timestamp,
                "session_id": session_id
            })
        
        # Emotional Memory - How the user feels about things
        self.emotional_memory[sentiment].append({
            "content": user_input,
            "timestamp": timestamp,
            "intensity": self._calculate_emotional_intensity(user_input),
            "topics": topics
        })
        
        # Update procedural memory (how to respond to patterns)
        self._update_procedural_memory(user_input, ai_response, sentiment, topics)
        
    def _calculate_importance(self, text, sentiment, topics):
        """Calculate importance score for memory consolidation"""
        score = len(topics) * 10  # More topics = more important
        if sentiment in ["very_positive", "very_negative"]:
            score += 20
        if len(text.split()) > 20:  # Longer questions often more important
            score += 15
        if any(word in text.lower() for word in ["meaning", "purpose", "life", "death", "god"]):
            score += 25
        return score
        
    def _calculate_emotional_intensity(self, text):
        """Calculate emotional intensity from text"""
        intense_words = ["very", "extremely", "deeply", "profound", "overwhelming", "desperate"]
        return sum(1 for word in intense_words if word in text.lower())
        
    def _update_procedural_memory(self, user_input, ai_response, sentiment, topics):
        """Learn patterns in how to respond"""
        pattern_key = f"{sentiment}_{','.join(sorted(topics))}"
        if pattern_key not in self.procedural_memory:
            self.procedural_memory[pattern_key] = {
                "successful_responses": [],
                "common_themes": defaultdict(int),
                "response_style": {}
            }
        
        # Store successful response patterns
        if len(ai_response) > 100:  # Assume longer responses are better
            self.procedural_memory[pattern_key]["successful_responses"].append({
                "response_length": len(ai_response),
                "verse_citations": ai_response.count("Chapter"),
                "timestamp": datetime.now()
            })
    
    def get_relevant_context(self, current_input, session_id, max_items=3):
        """Get relevant context from memory"""
        current_topics = extract_topics(current_input)
        current_sentiment = detect_sentiment(current_input)
        
        relevant_episodes = []
        
        # Get episodes from same session
        session_episodes = [ep for ep in self.episodic_memory if ep["session_id"] == session_id]
        relevant_episodes.extend(session_episodes[-2:])  # Last 2 from session
        
        # Get topically relevant episodes
        for episode in self.episodic_memory[-50:]:  # Check recent 50
            if any(topic in episode["topics"] for topic in current_topics):
                relevant_episodes.append(episode)
                
        # Sort by relevance and recency
        relevant_episodes.sort(key=lambda x: (
            x["importance_score"] + 
            (100 if x["session_id"] == session_id else 0) +
            (50 if x["sentiment"] == current_sentiment else 0)
        ), reverse=True)
        
        return relevant_episodes[:max_items]
    
    def get_personality_insights(self):
        """Analyze user personality from memory"""
        if not self.episodic_memory:
            return {}
            
        total_interactions = len(self.episodic_memory)
        
        # Emotional patterns
        emotion_counts = defaultdict(int)
        for sentiment, memories in self.emotional_memory.items():
            emotion_counts[sentiment] = len(memories)
        
        # Topic preferences
        topic_frequency = defaultdict(int)
        for episode in self.episodic_memory:
            for topic in episode["topics"]:
                topic_frequency[topic] += 1
        
        # Learning style analysis
        question_lengths = [len(ep["user_input"].split()) for ep in self.episodic_memory]
        avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
        
        return {
            "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral",
            "top_interests": sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
            "communication_style": "detailed" if avg_question_length > 15 else "concise",
            "engagement_level": total_interactions / max(1, (datetime.now() - self.episodic_memory[0]["timestamp"]).days or 1),
            "emotional_stability": 1 - (emotion_counts.get("negative", 0) / max(1, total_interactions)),
            "curiosity_level": sum(1 for ep in self.episodic_memory if "?" in ep["user_input"]) / max(1, total_interactions)
        }

# Initialize advanced memory manager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = AdvancedMemoryManager()

# ----------------------------- Advanced Features -----------------------------

def generate_session_id():
    """Generate unique session ID with timestamp"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

def create_new_session(title=None):
    """Create a new chat session with advanced features"""
    session_id = generate_session_id()
    if not title:
        # Generate smart title based on time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            title = f"Morning Contemplation {datetime.now().strftime('%m/%d')}"
        elif 12 <= hour < 17:
            title = f"Afternoon Reflection {datetime.now().strftime('%m/%d')}"
        elif 17 <= hour < 21:
            title = f"Evening Wisdom {datetime.now().strftime('%m/%d')}"
        else:
            title = f"Night Meditation {datetime.now().strftime('%m/%d')}"
    
    st.session_state.chat_sessions[session_id] = {
        "id": session_id,
        "title": title,
        "created_at": datetime.now(),
        "last_active": datetime.now(),
        "messages": [],
        "context_memory": [],
        "topics": set(),
        "sentiment_history": [],
        "verse_references": [],
        "breakthrough_moments": [],
        "wisdom_level": 1,
        "session_insights": [],
        "emotional_journey": [],
        "personal_growth_markers": []
    }
    st.session_state.current_session_id = session_id
    st.session_state.show_welcome = False
    return session_id

def detect_sentiment_advanced(text):
    """Advanced sentiment detection with intensity"""
    sentiments = {
        "very_positive": ["ecstatic", "blissful", "enlightened", "transcendent", "divine"],
        "positive": ["happy", "peace", "joy", "grateful", "blessed", "love", "hope", "content"],
        "neutral": ["think", "understand", "know", "question", "wonder"],
        "negative": ["sad", "angry", "fear", "anxious", "worried", "stress", "confused", "lost"],
        "very_negative": ["despair", "hopeless", "devastated", "broken", "empty"]
    }
    
    text_lower = text.lower()
    scores = {}
    
    for sentiment, words in sentiments.items():
        score = sum(2 if word in text_lower else 0 for word in words)
        # Boost score for intensity modifiers
        if any(mod in text_lower for mod in ["very", "extremely", "deeply", "so"]):
            score *= 1.5
        scores[sentiment] = score
    
    return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "neutral"

def extract_topics_advanced(text):
    """Advanced topic extraction with context awareness"""
    topic_keywords = {
        "karma_yoga": ["karma", "action", "duty", "work", "selfless", "service"],
        "bhakti_yoga": ["devotion", "love", "god", "worship", "surrender", "faith"],
        "raja_yoga": ["meditation", "focus", "concentration", "mind", "yoga", "practice"],
        "jnana_yoga": ["knowledge", "wisdom", "understanding", "truth", "self-realization"],
        "dharma": ["dharma", "righteousness", "moral", "ethics", "purpose", "duty"],
        "peace": ["peace", "calm", "tranquil", "serene", "stillness", "quiet"],
        "life_purpose": ["meaning", "purpose", "direction", "calling", "destiny"],
        "death_mortality": ["death", "mortality", "afterlife", "soul", "eternal"],
        "relationships": ["relationship", "family", "friend", "love", "conflict"],
        "suffering": ["pain", "suffering", "grief", "loss", "sorrow", "difficulty"],
        "success": ["success", "achievement", "goal", "ambition", "material"],
        "spirituality": ["spiritual", "divine", "sacred", "holy", "transcendent"]
    }
    
    text_lower = text.lower()
    topics = []
    
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            topics.append((topic, score))
    
    # Sort by relevance and return top topics
    topics.sort(key=lambda x: x[1], reverse=True)
    return [topic[0] for topic in topics[:3]]

def get_personalized_response_advanced(question, base_answer, context, session_id):
    """Advanced personalization with memory integration"""
    sentiment = detect_sentiment_advanced(question)
    topics = extract_topics_advanced(question)
    
    # Get relevant context from advanced memory
    relevant_context = st.session_state.memory_manager.get_relevant_context(
        question, session_id
    )
    
    # Update user profile with advanced analytics
    st.session_state.user_profile["total_questions"] += 1
    st.session_state.user_profile["wisdom_points"] += calculate_wisdom_points(question, sentiment, topics)
    st.session_state.user_profile["last_active"] = datetime.now()
    
    # Update preferred topics
    for topic in topics:
        st.session_state.user_profile["preferred_topics"][topic] += 1
    
    # Store interaction in advanced memory
    st.session_state.memory_manager.store_interaction(
        question, base_answer, context, sentiment, topics, session_id
    )
    
    # Get personality insights for response customization
    personality = st.session_state.memory_manager.get_personality_insights()
    
    # Enhance response based on personality and context
    enhanced_response = customize_response_style(
        base_answer, sentiment, topics, personality, relevant_context
    )
    
    # Check for achievements and breakthroughs
    achievement = check_achievements(question, sentiment, topics)
    breakthrough = detect_breakthrough_moment(question, relevant_context)
    
    if breakthrough:
        st.session_state.chat_sessions[session_id]["breakthrough_moments"].append({
            "timestamp": datetime.now(),
            "question": question,
            "insight": breakthrough
        })
    
    return enhanced_response, achievement, breakthrough

def calculate_wisdom_points(question, sentiment, topics):
    """Calculate wisdom points based on question quality and depth"""
    points = 10  # Base points
    
    # Bonus for deep topics
    deep_topics = ["life_purpose", "death_mortality", "jnana_yoga", "spirituality"]
    points += sum(15 for topic in topics if topic in deep_topics)
    
    # Bonus for question length and complexity
    word_count = len(question.split())
    if word_count > 20:
        points += 10
    if word_count > 50:
        points += 20
    
    # Bonus for philosophical questions
    if any(word in question.lower() for word in ["why", "how", "what is the meaning", "purpose"]):
        points += 15
    
    # Sentiment bonus
    if sentiment in ["very_positive", "very_negative"]:
        points += 10  # Deep emotions often lead to growth
    
    return points

def customize_response_style(base_answer, sentiment, topics, personality, context):
    """Customize response style based on user personality"""
    enhanced_response = base_answer
    
    # Adjust tone based on sentiment
    if sentiment in ["negative", "very_negative"]:
        enhanced_response += "\n\n💝 **Compassionate Guidance**: I sense you're navigating challenging waters. The Gita reminds us that even in our darkest moments, the light of wisdom within us never dims. Your courage to seek answers shows your spiritual strength."
    
    elif sentiment in ["positive", "very_positive"]:
        enhanced_response += "\n\n✨ **Joyful Reflection**: Your positive energy resonates beautifully with Krishna's teachings about finding divine joy in all experiences. This enthusiasm for spiritual growth is a precious gift."
    
    # Add personality-specific insights
    if personality and personality.get("communication_style") == "detailed":
        enhanced_response += "\n\n🔍 **Deeper Exploration**: Since you appreciate detailed discussions, consider how this teaching connects to the broader framework of the Gita's philosophy..."
    
    # Add contextual connections if relevant previous conversations exist
    if context and len(context) > 0:
        enhanced_response += f"\n\n🧵 **Connection to Your Journey**: I notice this builds beautifully on our previous conversation about {', '.join(set([c['topics'][0] if c['topics'] else 'spiritual growth' for c in context[:2]]))}. This shows your consistent dedication to understanding."
    
    return enhanced_response

def detect_breakthrough_moment(question, context):
    """Detect potential breakthrough moments in spiritual understanding"""
    breakthrough_indicators = [
        "suddenly understand", "it all makes sense", "revelation", "clarity",
        "never thought of it this way", "life-changing", "profound realization",
        "everything connected", "deeper meaning", "awakening"
    ]
    
    question_lower = question.lower()
    
    # Check for breakthrough language
    for indicator in breakthrough_indicators:
        if indicator in question_lower:
            return f"Breakthrough in understanding: {indicator}"
    
    # Check for pattern of increasing depth over time
    if context and len(context) > 2:
        recent_topics = []
        for c in context[-3:]:
            recent_topics.extend(c.get("topics", []))
        
        if len(set(recent_topics)) == 1 and len(recent_topics) > 2:
            return "Deep focused exploration of a single spiritual concept"
    
    return None

def check_achievements(question, sentiment, topics):
    """Check for unlockable achievements"""
    profile = st.session_state.user_profile
    
    achievements = {
        "first_question": profile["total_questions"] == 1,
        "seeker_milestone": profile["total_questions"] in [10, 25, 50, 100],
        "wisdom_gatherer": profile["wisdom_points"] >= 500,
        "deep_thinker": "jnana_yoga" in topics and profile["preferred_topics"]["jnana_yoga"] >= 5,
        "devoted_heart": "bhakti_yoga" in topics and profile["preferred_topics"]["bhakti_yoga"] >= 5,
        "karma_warrior": "karma_yoga" in topics and profile["preferred_topics"]["karma_yoga"] >= 5,
        "peaceful_soul": sentiment == "very_positive" and "peace" in topics,
        "resilient_spirit": sentiment in ["negative", "very_negative"] and profile["total_questions"] > 5,
        "consistent_seeker": profile["total_questions"] > 0 and (datetime.now() - profile["last_active"]).days <= 7,
        "wisdom_sage": profile["wisdom_points"] >= 1000,
        "enlightened_conversation": len(topics) >= 3 and any(topic in ["spirituality", "life_purpose"] for topic in topics)
    }
    
    for achievement, condition in achievements.items():
        if condition and achievement not in profile["achievements"]:
            profile["achievements"].append(achievement)
            return achievement
    
    return None

def get_gita_answer_advanced(question, session_context=[], personality_context={}):
    """Advanced QA with multi-layered context awareness"""
    if not client or not index:
        return "I apologize, but the AI service is currently unavailable. Please try again later.", ""
    
    try:
        # Build comprehensive context prompt
        context_prompt = ""
        
        # Session continuity
        if session_context:
            recent_context = session_context[-2:]  # More focused context
            context_prompt += "Recent conversation context:\n"
            for q, a in recent_context:
                context_prompt += f"User previously asked: {q[:100]}...\n"
                context_prompt += f"I responded about: {a[:150]}...\n"
            context_prompt += "\n"
        
        # Personality adaptation
        if personality_context:
            context_prompt += f"User's communication style: {personality_context.get('communication_style', 'balanced')}\n"
            context_prompt += f"Dominant interests: {', '.join([t[0] for t in personality_context.get('top_interests', [])])}\n"
            context_prompt += f"Emotional tendency: {personality_context.get('dominant_emotion', 'neutral')}\n\n"
        
        # Get relevant Gita verses
        query_vector = model.encode([question])
        D, I = index.search(np.array(query_vector), k=7)  # More context for better responses
        context = "\n".join([chunks[i] for i in I[0] if i < len(chunks)])
        
        # Advanced prompt engineering
        prompt = f"""You are an advanced AI spiritual guide with deep knowledge of the Bhagavad Gita and profound empathy for human spiritual journeys. You have perfect memory of previous conversations and can provide highly personalized guidance.

{context_prompt}

Relevant Bhagavad Gita verses and teachings:
{context}

Current question: {question}

Please provide a comprehensive, empathetic response that:
1. Directly addresses the question using specific Gita verses (include chapter and verse numbers when possible)
2. Shows awareness of conversation continuity and personal growth
3. Connects ancient wisdom to modern life applications
4. Uses a warm, wise, and personally engaging tone
5. Includes practical spiritual exercises or reflections when appropriate
6. Acknowledges the user's spiritual journey and progress

Format your response with clear sections and use emotive language that resonates with the human heart while maintaining philosophical depth.

Response:"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,  # Slightly higher for more creative responses
            max_tokens=600,   # More space for comprehensive answers
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip(), context
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question. Please try again. Error: {str(e)}", ""

def generate_smart_suggestions():
    """Generate contextual suggestions based on user's journey"""
    profile = st.session_state.user_profile
    
    base_suggestions = [
        "What is the meaning of dharma in daily life?",
        "How can I find inner peace amidst chaos?",
        "What does the Gita say about handling difficult relationships?",
        "How do I know if I'm on the right spiritual path?",
        "What is the difference between action and inaction?",
        "How can I overcome fear and anxiety according to the Gita?"
    ]
    
    # Personalized suggestions based on interests
    if "karma_yoga" in profile.get("preferred_topics", {}):
        base_suggestions.extend([
            "How can I practice selfless service in my work?",
            "What is the secret of working without attachment?"
        ])
    
    if "bhakti_yoga" in profile.get("preferred_topics", {}):
        base_suggestions.extend([
            "How do I cultivate pure devotion?",
            "What role does surrender play in spiritual growth?"
        ])
    
    # Return 6 random suggestions
    return random.sample(base_suggestions, min(6, len(base_suggestions)))

# ----------------------------- Main UI Components -----------------------------

def render_header():
    """Render the sophisticated header bar"""
    st.markdown(f"""
        <div class="header-bar">
            <div class="header-left">
                <div class="logo-container">
                    <div class="logo-icon">🕉️</div>
                    <div class="logo-text">Ask Scriptures AI</div>
                </div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    Advanced Intelligence Active
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the advanced sidebar with memory and analytics"""
    profile = st.session_state.user_profile
    
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <div class="sidebar-title">📚 Your Spiritual Journey</div>
            </div>
        """, unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("✨ New Sacred Conversation", key="new_chat_btn", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # Chat History with Enhanced Display
        st.markdown('<div class="chat-history-list">', unsafe_allow_html=True)
        
        sessions = sorted(
            st.session_state.chat_sessions.items(), 
            key=lambda x: x[1]["last_active"], 
            reverse=True
        )
        
        for session_id, session in sessions:
            is_active = session_id == st.session_state.current_session_id
            
            # Calculate session metrics
            time_diff = datetime.now() - session["created_at"]
            if time_diff.days > 0:
                time_ago = f"{time_diff.days}d ago"
            elif time_diff.seconds > 3600:
                time_ago = f"{time_diff.seconds // 3600}h ago"
            else:
                time_ago = f"{time_diff.seconds // 60}m ago"
            
            # Determine session sentiment
            sentiments = session.get("sentiment_history", [])
            dominant_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
            
            session_button_html = f"""
                <div class="chat-history-item {'active' if is_active else ''}" onclick="selectSession('{session_id}')">
                    <div class="chat-history-title">{session['title']}</div>
                    <div class="chat-history-meta">
                        <span>🕐 {time_ago} • {len(session['messages'])} msgs</span>
                        <span class="chat-sentiment sentiment-{dominant_sentiment}">
                            {dominant_sentiment}
                        </span>
                    </div>
                    {f'<div class="memory-indicator">🧠</div>' if len(session.get('breakthrough_moments', [])) > 0 else ''}
                </div>
            """
            
            if st.button(
                f"💬 {session['title']}\n🕐 {time_ago} • {len(session['messages'])} messages",
                key=f"session_{session_id}",
                use_container_width=True
            ):
                st.session_state.current_session_id = session_id
                st.session_state.show_welcome = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced User Profile Card
        wisdom_level = min(10, profile["wisdom_points"] // 100 + 1)
        level_progress = (profile["wisdom_points"] % 100)
        
        personality = st.session_state.memory_manager.get_personality_insights()
        
        st.markdown(f"""
            <div class="profile-card">
                <div class="profile-header">
                    <div class="profile-avatar">
                        🧘‍♂️
                        <div class="memory-indicator">{len(st.session_state.memory_manager.episodic_memory)}</div>
                    </div>
                    <div class="profile-info">
                        <div class="profile-name">Spiritual Seeker</div>
                        <div class="profile-level">Wisdom Level {wisdom_level}</div>
                        <div class="level-progress">
                            <div class="level-progress-fill" style="width: {level_progress}%"></div>
                        </div>
                    </div>
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{profile["total_questions"]}</div>
                        <div class="stat-label">Questions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{profile["wisdom_points"]}</div>
                        <div class="stat-label">Wisdom Pts</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(st.session_state.chat_sessions)}</div>
                        <div class="stat-label">Sessions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(profile["achievements"])}</div>
                        <div class="stat-label">Achievements</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Achievements Section
        if profile["achievements"]:
            achievement_badges = ""
            for achievement in profile["achievements"][-6:]:  # Show last 6 achievements
                badge_name = achievement.replace("_", " ").title()
                achievement_badges += f'<div class="achievement-badge">🏆 {badge_name}</div>'
            
            st.markdown(f"""
                <div class="achievements-section">
                    <div class="achievements-title">🏆 Recent Achievements</div>
                    <div class="achievement-badges">
                        {achievement_badges}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Personality Insights (if available)
        if personality and len(st.session_state.memory_manager.episodic_memory) > 5:
            top_interests = personality.get("top_interests", [])[:3]
            
            insights_html = "<div class='insights-section'>"
            insights_html += "<div class='sidebar-title'>🎯 Your Spiritual Profile</div>"
            
            if personality.get("dominant_emotion") != "neutral":
                insights_html += f"<div class='insight-item'>💙 Emotional tone: {personality['dominant_emotion'].replace('_', ' ').title()}</div>"
            
            if top_interests:
                interests_str = ", ".join([interest[0].replace("_", " ").title() for interest in top_interests])
                insights_html += f"<div class='insight-item'>🌟 Key interests: {interests_str}</div>"
            
            if personality.get("communication_style"):
                insights_html += f"<div class='insight-item'>💬 Style: {personality['communication_style'].title()}</div>"
            
            insights_html += "</div>"
            st.markdown(insights_html, unsafe_allow_html=True)

def render_welcome_screen():
    """Render the sophisticated welcome screen"""
    suggestions = generate_smart_suggestions()
    
    st.markdown(f"""
        <div class="welcome-container">
            <div class="welcome-title">Welcome to Your Spiritual Journey</div>
            <div class="welcome-subtitle">
                Discover ancient wisdom through advanced AI that remembers, learns, and grows with you.
                Every conversation deepens our connection and understanding.
            </div>
            
            <div class="feature-grid">
                <div class="feature-card" onclick="startConversation('memory')">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-title">Advanced Memory</div>
                    <div class="feature-desc">I remember our entire journey together, building deeper understanding over time</div>
                </div>
                
                <div class="feature-card" onclick="startConversation('personality')">
                    <div class="feature-icon">🎭</div>
                    <div class="feature-title">Personality Adaptation</div>
                    <div class="feature-desc">Responses tailored to your unique spiritual style and emotional patterns</div>
                </div>
                
                <div class="feature-card" onclick="startConversation('breakthrough')">
                    <div class="feature-icon">💡</div>
                    <div class="feature-title">Breakthrough Detection</div>
                    <div class="feature-desc">Recognition of your spiritual insights and growth moments</div>
                </div>
                
                <div class="feature-card" onclick="startConversation('context')">
                    <div class="feature-icon">🧵</div>
                    <div class="feature-title">Contextual Wisdom</div>
                    <div class="feature-desc">Connections across conversations for holistic spiritual guidance</div>
                </div>
            </div>
            
            <div class="suggestions-container">
    """, unsafe_allow_html=True)
    
    # Render suggestion pills
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion[:20]}", use_container_width=False):
            st.session_state.current_question = suggestion
            if not st.session_state.current_session_id:
                create_new_session()
            st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_chat_interface():
    """Render the main chat interface using native chat components."""
    if not st.session_state.current_session_id:
        return

    session = st.session_state.chat_sessions[st.session_state.current_session_id]

    for i, (role, message, timestamp, metadata) in enumerate(session["messages"]):
        render_message(role, message, timestamp, metadata, i)

    # Thinking indicator
    if st.session_state.get("thinking_animation", False):
        with st.chat_message("assistant"):
            st.markdown("""
                <div class="thinking-indicator">
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                    <div class="thinking-text">Contemplating the depths of wisdom...</div>
                </div>
            """, unsafe_allow_html=True)


def render_message(role, message, timestamp, metadata, index):
    """Render chat message with native Streamlit chat API + custom CSS (HTML not escaped)."""
    time_str = timestamp.strftime("%H:%M")

    session = st.session_state.chat_sessions[st.session_state.current_session_id]
    is_breakthrough = any(
        breakthrough["timestamp"].strftime("%H:%M") == time_str
        for breakthrough in session.get("breakthrough_moments", [])
    )

    sentiment = metadata.get("sentiment", "neutral")
    topics = metadata.get("topics", [])

    wrapper_class = "message-wrapper"
    if is_breakthrough:
        wrapper_class += " breakthrough-moment"

    html_block = f"""
    <div class="{wrapper_class}">
        <div class="message-header">
            <div class="message-author">
                {"Ask Scriptures AI" if role == "assistant" else "You"}
            </div>
            <div class="message-meta">
                <span class="message-time">{time_str}</span>
                {f'<span class="message-sentiment sentiment-{sentiment}">{sentiment}</span>' if sentiment != "neutral" else ''}
                {f'<span class="message-topics">🏷️ {", ".join(topics[:2])}</span>' if topics else ''}
            </div>
        </div>
        <div class="message-text">{message}</div>
    </div>
    """

    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(html_block, unsafe_allow_html=True)


def render_input_area():
    """Render the advanced input area"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    
    # Chat input with placeholder that adapts to user's journey
    profile = st.session_state.user_profile
    
    placeholders = [
        "Share what's on your heart and mind...",
        "What spiritual wisdom do you seek today?",
        "How can the Gita guide you right now?",
        "What questions arise from your spiritual journey?"
    ]
    
    if profile["total_questions"] > 10:
        placeholders.extend([
            "Let's dive deeper into your spiritual exploration...",
            "What new insights have emerged for you?",
            "How has your understanding evolved?"
        ])
    
    placeholder = random.choice(placeholders)
    
    # Handle preset question from suggestions
    initial_value = ""
    if hasattr(st.session_state, 'current_question'):
        initial_value = st.session_state.current_question
        del st.session_state.current_question
    
    user_input = st.chat_input(
        placeholder=placeholder,
        key="main_chat_input"
    )
    
    if initial_value:
        user_input = initial_value
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input

def handle_user_input(user_input):
    """Handle user input with advanced processing"""
    if not user_input:
        return
    
    # Ensure we have a session
    if not st.session_state.current_session_id:
        create_new_session()
    
    session_id = st.session_state.current_session_id
    session = st.session_state.chat_sessions[session_id]
    
    # Update session activity
    session["last_active"] = datetime.now()
    
    # Show thinking animation
    st.session_state.thinking_animation = True
    
    # Add user message
    timestamp = datetime.now()
    sentiment = detect_sentiment_advanced(user_input)
    topics = extract_topics_advanced(user_input)
    
    user_metadata = {
        "sentiment": sentiment,
        "topics": topics,
        "word_count": len(user_input.split()),
        "contextual": len(session["messages"]) > 0
    }
    
    session["messages"].append(("user", user_input, timestamp, user_metadata))
    session["sentiment_history"].append(sentiment)
    session["topics"].update(topics)
    
    # Get context for AI response
    recent_context = [(msg[1], msg[1]) for msg in session["messages"][-4:] if msg[0] == "user"]
    personality_context = st.session_state.memory_manager.get_personality_insights()
    
    try:
        # Generate AI response
        base_answer, context = get_gita_answer_advanced(
            user_input, recent_context, personality_context
        )
        
        # Apply advanced personalization
        enhanced_answer, achievement, breakthrough = get_personalized_response_advanced(
            user_input, base_answer, context, session_id
        )
        
        # Add AI message
        ai_metadata = {
            "sentiment": sentiment,
            "topics": topics,
            "context_used": len(context) > 0,
            "achievement": achievement,
            "breakthrough": breakthrough
        }
        
        session["messages"].append(("assistant", enhanced_answer, datetime.now(), ai_metadata))
        
        # Log to Google Sheets
        append_chat_to_sheet(session_id, user_input, enhanced_answer, context, sentiment, topics)
        
        # Handle achievement popup
        if achievement:
            st.session_state.achievement_popup = {
                "achievement": achievement,
                "timestamp": datetime.now()
            }
        
        # Update session title if it's the first message
        if len(session["messages"]) == 2:  # First user message + first AI response
            session["title"] = generate_session_title(user_input, topics)
        
    except Exception as e:
        error_message = f"I apologize, but I encountered an error while processing your question. Please try again. Error: {str(e)}"
        session["messages"].append(("assistant", error_message, datetime.now(), {"error": True}))
    
    finally:
        st.session_state.thinking_animation = False

def generate_session_title(first_question, topics):
    """Generate intelligent session titles"""
    titles = {
        "karma_yoga": ["Path of Action", "Sacred Service", "Selfless Work"],
        "bhakti_yoga": ["Path of Devotion", "Divine Love", "Surrender & Faith"],
        "raja_yoga": ["Inner Journey", "Meditation Mastery", "Mind & Focus"],
        "jnana_yoga": ["Wisdom Seeking", "Self-Knowledge", "Truth Inquiry"],
        "dharma": ["Righteous Path", "Life Purpose", "Sacred Duty"],
        "peace": ["Inner Peace", "Tranquility", "Calm Mind"],
        "life_purpose": ["Life's Meaning", "Soul Purpose", "Divine Calling"],
        "relationships": ["Sacred Bonds", "Love & Connection", "Relationship Wisdom"],
        "suffering": ["Through Difficulty", "Pain to Growth", "Healing Journey"],
        "spirituality": ["Divine Connection", "Sacred Quest", "Spiritual Awakening"]
    }
    
    if topics:
        primary_topic = topics[0]
        if primary_topic in titles:
            return random.choice(titles[primary_topic])
    
    # Fallback: use first few words of question
    words = first_question.split()[:4]
    return " ".join(words).capitalize()

def render_achievement_popup():
    """Render achievement popup notification"""
    if st.session_state.achievement_popup:
        achievement = st.session_state.achievement_popup["achievement"]
        
        achievement_data = {
            "first_question": {"icon": "🌟", "title": "First Steps", "desc": "Welcome to your spiritual journey!"},
            "seeker_milestone": {"icon": "🏆", "title": "Dedicated Seeker", "desc": "Your commitment to growth shines!"},
            "wisdom_gatherer": {"icon": "💎", "title": "Wisdom Gatherer", "desc": "You've accumulated substantial spiritual insights!"},
            "deep_thinker": {"icon": "🤔", "title": "Jnana Yogi", "desc": "Your love for wisdom and knowledge is evident!"},
            "devoted_heart": {"icon": "❤️", "title": "Bhakti Yogi", "desc": "Your heart overflows with devotion!"},
            "karma_warrior": {"icon": "⚔️", "title": "Karma Yogi", "desc": "You understand the path of selfless action!"},
            "peaceful_soul": {"icon": "🕊️", "title": "Peaceful Soul", "desc": "You radiate inner tranquility!"},
            "resilient_spirit": {"icon": "💪", "title": "Resilient Spirit", "desc": "Your strength through challenges is inspiring!"},
            "wisdom_sage": {"icon": "🧙‍♂️", "title": "Wisdom Sage", "desc": "You've reached profound spiritual understanding!"}
        }
        
        data = achievement_data.get(achievement, {"icon": "🌟", "title": "Achievement", "desc": "Well done!"})
        
        st.markdown(f"""
            <div class="achievement-toast">
                <div class="achievement-header">
                    <div class="achievement-icon">{data["icon"]}</div>
                    <div class="achievement-title">{data["title"]}</div>
                </div>
                <div class="achievement-desc">{data["desc"]}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Auto-hide after 4 seconds
        time.sleep(4)
        st.session_state.achievement_popup = None

# ----------------------------- Main Application Flow -----------------------------

def main():
    """Main application entry point"""
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.show_welcome or not st.session_state.current_session_id:
        render_welcome_screen()
    else:
        render_chat_interface()
    
    # Input area (always visible)
    user_input = render_input_area()
    
    # Handle user input
    if user_input:
        handle_user_input(user_input)
        st.rerun()
    
    # Render achievement popup if needed
    if st.session_state.achievement_popup:
        render_achievement_popup()

# ----------------------------- Application Initialization -----------------------------

if __name__ == "__main__":
    # Initialize session if none exists
    if not st.session_state.current_session_id and st.session_state.chat_sessions:
        # Set most recent session as current
        latest_session = max(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]["last_active"]
        )
        st.session_state.current_session_id = latest_session[0]
        st.session_state.show_welcome = False
    
    # Run main application
    main()

# ----------------------------- Additional Utility Functions -----------------------------

def export_conversation_history():
    """Export conversation history for user"""
    if not st.session_state.chat_sessions:
        return None
    
    export_data = {
        "user_profile": st.session_state.user_profile,
        "sessions": {},
        "memory_insights": st.session_state.memory_manager.get_personality_insights(),
        "export_timestamp": datetime.now().isoformat()
    }
    
    for session_id, session in st.session_state.chat_sessions.items():
        export_data["sessions"][session_id] = {
            "title": session["title"],
            "created_at": session["created_at"].isoformat(),
            "messages": [(role, msg, ts.isoformat(), meta) for role, msg, ts, meta in session["messages"]],
            "breakthrough_moments": [
                {**moment, "timestamp": moment["timestamp"].isoformat()} 
                for moment in session.get("breakthrough_moments", [])
            ]
        }
    
    return json.dumps(export_data, indent=2)

def get_spiritual_analytics():
    """Get comprehensive spiritual journey analytics"""
    profile = st.session_state.user_profile
    memory = st.session_state.memory_manager
    personality = memory.get_personality_insights()
    
    analytics = {
        "journey_overview": {
            "total_questions": profile["total_questions"],
            "wisdom_points": profile["wisdom_points"],
            "sessions_count": len(st.session_state.chat_sessions),
            "achievements_unlocked": len(profile["achievements"]),
            "spiritual_level": min(10, profile["wisdom_points"] // 100 + 1)
        },
        "personality_insights": personality,
        "growth_patterns": {
            "topic_evolution": dict(profile.get("preferred_topics", {})),
            "emotional_journey": [
                {"sentiment": k, "frequency": len(v)} 
                for k, v in memory.emotional_memory.items()
            ],
            "breakthrough_count": sum(
                len(session.get("breakthrough_moments", [])) 
                for session in st.session_state.chat_sessions.values()
            )
        },
        "recommendations": generate_spiritual_recommendations(personality, profile)
    }
    
    return analytics

def generate_spiritual_recommendations(personality, profile):
    """Generate personalized spiritual practice recommendations"""
    recommendations = []
    
    if personality.get("dominant_emotion") == "negative":
        recommendations.append({
            "type": "emotional_healing",
            "title": "Healing Practices",
            "suggestion": "Focus on Gita chapters 12 and 18 for emotional peace and surrender practices"
        })
    
    top_interests = personality.get("top_interests", [])
    if top_interests:
        primary_interest = top_interests[0][0]
        
        practice_map = {
            "karma_yoga": "Daily selfless service practice - dedicate actions to the divine",
            "bhakti_yoga": "Heart-centered devotional practices - chanting and prayer",
            "raja_yoga": "Regular meditation and breath awareness practices",
            "jnana_yoga": "Study of spiritual texts and self-inquiry practices"
        }
        
        if primary_interest in practice_map:
            recommendations.append({
                "type": "spiritual_practice",
                "title": f"{primary_interest.replace('_', ' ').title()} Focus",
                "suggestion": practice_map[primary_interest]
            })
    
    if profile["wisdom_points"] > 500:
        recommendations.append({
            "type": "advanced_study",
            "title": "Advanced Learning",
            "suggestion": "Explore commentaries by great masters like Shankara, Ramanuja, or contemporary teachers"
        })
    
    return recommendations

# Add CSS for enhanced mobile responsiveness
# Add CSS for enhanced mobile responsiveness
st.markdown("""
    <style>
    @media (max-width: 480px) {
        .welcome-title { font-size: 2rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: 1fr 1fr; gap: 0.5rem; }
        .message-wrapper { gap: 0.5rem; margin-bottom: 1.5rem; }
        .input-container { padding: 1rem 0.5rem; }
        .header-bar { padding: 0.75rem; }
        .logo-text { font-size: 1rem; }
        .status-indicator { display: none; }
    }
    
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: #000000;
        padding: 0;
    }
    
    .block-container {
        max-width: 100%;
        padding: 0;
        margin: 0;
    }
    
    /* Sidebar Styling */
    .css-1y4p8pa {
        width: 320px !important;
        background: #0a0a0a !important;
        border-right: 1px solid #1a1a1a !important;
    }
    
    .css-1y4p8pa .block-container {
        padding: 1rem !important;
    }
    
    /* Main Chat Area */
    .main-chat-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        max-width: 900px;
        margin: 0 auto;
        position: relative;
    }
    
    /* Header Bar */
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #1a1a1a;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(10px);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 0 30px rgba(78, 205, 196, 0.4); }
    }
    
    .logo-text {
        font-size: 1.25rem;
        font-weight: 600;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: rgba(78, 205, 196, 0.1);
        border: 1px solid rgba(78, 205, 196, 0.3);
        border-radius: 20px;
        font-size: 12px;
        color: #4ECDC4;
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: #4ECDC4;
        border-radius: 50%;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    /* Sidebar Content */
    .sidebar-header {
        padding: 1.5rem 1rem;
        border-bottom: 1px solid #1a1a1a;
    }
    
    .sidebar-title {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .new-chat-btn {
        width: 100%;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: #ffffff;
        border: none;
        padding: 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 14px;
        position: relative;
        overflow: hidden;
    }
    
    .new-chat-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .new-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
    }
    
    .new-chat-btn:hover::before {
        left: 100%;
    }
    
    /* Chat History List */
    .chat-history-list {
        padding: 1rem 0;
        overflow-y: auto;
        max-height: calc(100vh - 500px);
    }
    
    .chat-history-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0.5rem;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
        position: relative;
        background: rgba(10, 10, 10, 0.5);
    }
    
    .chat-history-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s;
        border-radius: 12px;
    }
    
    .chat-history-item:hover {
        background: #111111;
        border-color: #2a2a2a;
        transform: translateX(4px);
    }
    
    .chat-history-item:hover::before {
        opacity: 1;
    }
    
    .chat-history-item.active {
        background: rgba(255, 107, 107, 0.15);
        border-color: #FF6B6B;
        transform: translateX(4px);
    }
    
    .chat-history-title {
        font-size: 14px;
        color: #e4e4e4;
        font-weight: 500;
        margin-bottom: 0.25rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        position: relative;
        z-index: 1;
    }
    
    .chat-history-meta {
        font-size: 12px;
        color: #666;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        z-index: 1;
    }
    
    .chat-sentiment {
        padding: 0.125rem 0.5rem;
        border-radius: 10px;
        font-size: 10px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .sentiment-positive { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
    .sentiment-negative { background: rgba(244, 67, 54, 0.2); color: #f44336; }
    .sentiment-neutral { background: rgba(158, 158, 158, 0.2); color: #9e9e9e; }
    
    /* User Profile Card */
    .profile-card {
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(26, 26, 26, 0.6) 100%);
        border: 1px solid #1a1a1a;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .profile-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    }
    
    .profile-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .profile-avatar {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        position: relative;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .profile-avatar::after {
        content: '';
        position: absolute;
        inset: -2px;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 50%;
        z-index: -1;
        animation: rotate 8s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .profile-info {
        flex: 1;
    }
    
    .profile-name {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .profile-level {
        font-size: 13px;
        color: #888;
        margin-bottom: 4px;
    }
    
    .level-progress {
        width: 100%;
        height: 4px;
        background: #222;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 4px;
    }
    
    .level-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        width: 75%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(17, 17, 17, 0.8);
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(42, 42, 42, 0.5);
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .stat-item:hover {
        border-color: #FF6B6B;
        background: rgba(255, 107, 107, 0.05);
    }
    
    .stat-value {
        font-size: 20px;
        font-weight: 700;
        color: #FF6B6B;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-label {
        font-size: 11px;
        color: #666;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Achievements Section */
    .achievements-section {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(10, 10, 10, 0.6);
        border-radius: 12px;
        border: 1px solid #1a1a1a;
    }
    
    .achievements-title {
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .achievement-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .achievement-badge {
        padding: 0.25rem 0.5rem;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(78, 205, 196, 0.2) 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 12px;
        font-size: 10px;
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* Chat Messages */
    .chat-messages-container {
        flex: 1;
        overflow-y: auto;
        padding: 2rem 1.5rem 8rem;
        scroll-behavior: smooth;
    }
    
    .message-wrapper {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-top: 2px;
        position: relative;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .ai-avatar {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
        animation: aiGlow 3s ease-in-out infinite;
    }
    
    @keyframes aiGlow {
        0%, 100% { box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(78, 205, 196, 0.4); }
    }
    
    .message-content {
        flex: 1;
        color: #e4e4e4;
        font-size: 15px;
        line-height: 1.7;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .message-author {
        font-weight: 600;
        font-size: 14px;
        color: #ffffff;
    }
    
    .message-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 12px;
        color: #666;
    }
    
    .message-time {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .message-sentiment {
        padding: 0.125rem 0.5rem;
        border-radius: 8px;
        font-size: 10px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .message-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        color: #e4e4e4;
    }
    
    /* Enhanced Message Types */
    .message-ai .message-text {
        background: rgba(10, 10, 10, 0.4);
        padding: 1rem;
        border-radius: 12px;
        border-left: 3px solid #FF6B6B;
        margin-top: 0.5rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    }
    
    .insight-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .insight-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .insight-title {
        font-size: 14px;
        font-weight: 600;
        color: #FF6B6B;
        flex: 1;
    }
    
    .insight-confidence {
        font-size: 11px;
        color: #4ECDC4;
        background: rgba(78, 205, 196, 0.1);
        padding: 0.25rem 0.5rem;
        border-radius: 10px;
    }
    
    .insight-content {
        font-size: 14px;
        color: #e4e4e4;
        line-height: 1.6;
    }
    
    /* Verse Citation */
    .verse-citation {
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(26, 26, 26, 0.6) 100%);
        border-left: 4px solid #FF6B6B;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        position: relative;
    }
    
    .verse-citation::before {
        content: '"';
        position: absolute;
        top: -10px;
        left: 10px;
        font-size: 4rem;
        color: rgba(255, 107, 107, 0.2);
        font-family: serif;
    }
    
    .verse-text {
        font-style: italic;
        color: #e4e4e4;
        margin-bottom: 0.75rem;
        font-size: 16px;
        line-height: 1.8;
        position: relative;
        z-index: 1;
    }
    
    .verse-reference {
        font-size: 13px;
        color: #4ECDC4;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Welcome Screen */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        max-width: 800px;
        margin: 0 auto;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .welcome-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: titleGlow 4s ease-in-out infinite;
        letter-spacing: -0.02em;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: brightness(1) saturate(1); }
        50% { filter: brightness(1.2) saturate(1.3); }
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 3rem;
        font-weight: 300;
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 4rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(26, 26, 26, 0.6) 100%);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.05) 0%, rgba(78, 205, 196, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .feature-card:hover {
        border-color: #FF6B6B;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(255, 107, 107, 0.2);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        position: relative;
        z-index: 1;
    }
    
    .feature-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }
    
    .feature-desc {
        font-size: 14px;
        color: #888;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    /* Suggestion Pills */
    .suggestions-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 2rem;
        justify-content: center;
    }
    
    .suggestion-pill {
        background: rgba(10, 10, 10, 0.8);
        border: 1px solid #2a2a2a;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 14px;
        color: #888;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .suggestion-pill::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s;
        border-radius: 25px;
    }
    
    .suggestion-pill:hover {
        color: #ffffff;
        border-color: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.2);
    }
    
    .suggestion-pill:hover::before {
        opacity: 1;
    }
    
    /* Input Container */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 900px;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(20px);
        border-top: 1px solid #1a1a1a;
        padding: 1.5rem 2rem;
        z-index: 100;
    }
    
    .input-wrapper {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: rgba(10, 10, 10, 0.8);
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 0.75rem 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .input-wrapper:focus-within {
        border-color: #FF6B6B;
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.2);
        background: rgba(15, 15, 15, 0.9);
    }
    
    .message-input {
        flex: 1;
        background: none;
        border: none;
        outline: none;
        color: #ffffff;
        font-size: 15px;
        line-height: 1.5;
        font-family: inherit;
        resize: none;
        min-height: 24px;
        max-height: 120px;
    }
    
    .message-input::placeholder {
        color: #666;
    }
    
    .send-button {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        border: none;
        border-radius: 12px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        flex-shrink: 0;
    }
    
    .send-button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
    }
    
    .send-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.3s ease-out;
    }
    
    .typing-dots {
        display: flex;
        gap: 0.25rem;
        padding: 1rem;
        background: rgba(10, 10, 10, 0.4);
        border-radius: 12px;
        border-left: 3px solid #FF6B6B;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #FF6B6B;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typingBounce {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111111;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 4px;
        transition: background 0.3s;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #e55a5a 0%, #45b8b0 100%);
    }
    
    /* Loading States */
    .loading-shimmer {
        background: linear-gradient(90deg, 
            rgba(255, 255, 255, 0.1) 25%, 
            rgba(255, 255, 255, 0.2) 50%, 
            rgba(255, 255, 255, 0.1) 75%
        );
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .skeleton-text {
        height: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    
    .skeleton-text.short { width: 60%; }
    .skeleton-text.medium { width: 80%; }
    .skeleton-text.long { width: 100%; }
    
    /* Error States */
    .error-message {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(229, 57, 53, 0.1) 100%);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #f44336;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .error-icon {
        width: 24px;
        height: 24px;
        background: #f44336;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: #ffffff;
        flex-shrink: 0;
    }
    
    /* Success States */
    .success-message {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(67, 160, 71, 0.1) 100%);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #4CAF50;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .success-icon {
        width: 24px;
        height: 24px;
        background: #4CAF50;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: #ffffff;
        flex-shrink: 0;
    }
    
    /* Modal Overlay */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-content {
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.95) 0%, rgba(26, 26, 26, 0.95) 100%);
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 2rem;
        max-width: 500px;
        width: 100%;
        max-height: 80vh;
        overflow-y: auto;
        animation: modalSlide 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    @keyframes modalSlide {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .modal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .modal-title {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .modal-close {
        background: rgba(42, 42, 42, 0.8);
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #888;
        transition: all 0.3s;
    }
    
    .modal-close:hover {
        background: rgba(244, 67, 54, 0.2);
        border-color: #f44336;
        color: #f44336;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: #ffffff;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s;
        z-index: 1000;
        margin-bottom: 5px;
    }
    
    .tooltip::before {
        content: '';
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 5px solid transparent;
        border-top-color: rgba(0, 0, 0, 0.9);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s;
        z-index: 1000;
    }
    
    .tooltip:hover::after,
    .tooltip:hover::before {
        opacity: 1;
        visibility: visible;
    }
    
    /* Context Menu */
    .context-menu {
        position: fixed;
        background: rgba(10, 10, 10, 0.95);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 0.5rem 0;
        min-width: 180px;
        z-index: 1000;
        backdrop-filter: blur(20px);
        animation: contextMenuSlide 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    @keyframes contextMenuSlide {
        from {
            opacity: 0;
            transform: translateY(-10px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .context-menu-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        font-size: 14px;
        color: #e4e4e4;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .context-menu-item:hover {
        background: rgba(255, 107, 107, 0.1);
        color: #FF6B6B;
    }
    
    .context-menu-item.danger:hover {
        background: rgba(244, 67, 54, 0.1);
        color: #f44336;
    }
    
    .context-menu-divider {
        height: 1px;
        background: #2a2a2a;
        margin: 0.25rem 0;
    }
    
    /* Responsive Design Enhancements */
    @media (max-width: 768px) {
        .main-chat-container { max-width: 100%; }
        .header-bar { padding: 1rem; }
        .chat-messages-container { padding: 1.5rem 1rem 8rem; }
        .input-container { padding: 1rem; }
        .welcome-container { padding: 3rem 1.5rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: 1fr 1fr; }
        .message-wrapper { gap: 0.75rem; margin-bottom: 1.5rem; }
        .profile-card { margin: 1rem 0; }
        .modal-content { margin: 1rem; padding: 1.5rem; }
        .suggestions-container { justify-content: flex-start; }
    }
    
    @media (max-width: 480px) {
        .welcome-title { font-size: 2rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: 1fr 1fr; gap: 0.5rem; }
        .message-wrapper { gap: 0.5rem; margin-bottom: 1.5rem; }
        .input-container { padding: 1rem 0.5rem; }
        .header-bar { padding: 0.75rem; }
        .logo-text { font-size: 1rem; }
        .status-indicator { display: none; }
        .css-1y4p8pa { width: 280px !important; }
        .profile-avatar { width: 48px; height: 48px; font-size: 24px; }
        .modal-content { padding: 1.25rem; }
        .context-menu { min-width: 150px; }
    }
    
    /* Print Styles */
    @media print {
        .sidebar-header,
        .input-container,
        .header-bar,
        .context-menu,
        .modal-overlay {
            display: none !important;
        }
        
        .main-chat-container {
            height: auto;
            max-width: none;
        }
        
        .chat-messages-container {
            padding: 1rem;
            overflow: visible;
        }
        
        .message-wrapper {
            break-inside: avoid;
        }
        
        body {
            background: white !important;
            color: black !important;
        }
    }
    
    /* Animation Performance Optimization */
    .message-wrapper,
    .feature-card,
    .suggestion-pill,
    .chat-history-item,
    .new-chat-btn,
    .send-button {
        will-change: transform;
    }
    
    /* Focus Styles for Accessibility */
    .new-chat-btn:focus,
    .send-button:focus,
    .suggestion-pill:focus,
    .chat-history-item:focus {
        outline: 2px solid #FF6B6B;
        outline-offset: 2px;
    }
    
    .message-input:focus {
        outline: none;
    }
    
    /* High Contrast Mode Support */
    @media (prefers-contrast: high) {
        .stApp {
            background: #000000;
            color: #ffffff;
        }
        
        .message-text,
        .verse-text,
        .insight-content {
            color: #ffffff;
        }
        
        .border,
        .chat-history-item,
        .profile-card {
            border-color: #ffffff;
        }
    }
    
    /* Reduced Motion Support */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        
        .pulse,
        .rotate,
        .shimmer,
        .blink,
        .aiGlow,
        .titleGlow {
            animation: none !important;
        }
    }
    
    </style>
""", unsafe_allow_html=True)

print("🕉️ Ask Scriptures AI - Advanced Spiritual Intelligence System Loaded Successfully!")
print("✨ Features: Advanced Memory | Personality Adaptation | Breakthrough Detection | Contextual Wisdom")
print("🧠 Ready to provide personalized spiritual guidance with complete conversation history awareness.")





import streamlit as st
import pickle
from datetime import datetime

# --- Configuration for Black & White Theme (Dark Mode) ---
st.set_page_config(
    page_title="Language Detection Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Monochrome Aesthetics ---
st.markdown("""
    <style>
    /* Main Streamlit container adjustments */
    .stApp {
        background-color: #1E1E1E; /* Dark Grey/Black background */
        color: #F0F0F0; /* Light Grey/White text */
    }

    /* Target the text area and input fields for a clean look */
    .stTextArea, .stTextInput {
        background-color: #2D2D2D !important; /* Slightly lighter than background */
        border: 1px solid #444444 !important;
        color: #F0F0F0 !important;
    }
    
    /* Enhance the button to look modern and monochrome */
    .stButton>button {
        background-color: #1E1E1E;
        color: #F0F0F0;
        border: 2px solid #F0F0F0; /* White border for contrast */
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #444444; /* Darker hover effect */
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
    }
    
    /* Customizing success/info boxes to align with the theme */
    div[data-testid="stSuccess"] {
        background-color: #333333; /* Darker background for success */
        color: #00FF00; /* Use a bright color for the actual text feedback */
        border-left: 5px solid #00FF00;
    }
    div[data-testid="stInfo"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border-left: 5px solid #FFFFFF;
    }
    div[data-testid="stWarning"] {
        background-color: #333333;
        color: #FFFF00;
        border-left: 5px solid #FFFF00;
    }

    </style>
    """, unsafe_allow_html=True)


# --- Load Model & Vectorizer ---

# Use st.cache_resource to load large objects once for better performance
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open("language_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        st.error("🚨 *Error:* Model or Vectorizer file not found. Please ensure language_model.pkl and vectorizer.pkl are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 *Error loading resources:* {e}")
        st.stop()

model, vectorizer = load_resources()


# --- Rule-based responses for 22 languages ---
responses = {
    "English": "Hello! How can I help you?",
    "Hindi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "French": "Bonjour ! Comment puis-je vous aider ?",
    "Spanish": "¡Hola! ¿Cómo puedo ayudarte?",
    "German": "Hallo! Wie kann ich Ihnen helfen?",
    "Arabic": "مرحباً! كيف يمكنني مساعدتك؟",
    "Russian": "Здравствуйте! Чем могу помочь?",
    "Chinese": "你好！我可以帮你什么？",
    "Japanese": "こんにちは！どのようにお手伝いできますか？",
    "Korean": "안녕하세요! 어떻게 도와드릴까요?",
    "Portuguese": "Olá! Como posso ajudar?",
    "Italian": "Ciao! Come posso aiutarti?",
    "Turkish": "Merhaba! Size nasıl yardımcı olabilirim?",
    "Tamil": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவலாம்?",
    "Urdu": "السلام علیکم! میں آپ کی کیسے مدد کر سکتا ہوں؟",
    "Thai": "สวัสดี! ฉันช่วยคุณได้อย่างไร?",
    "Dutch": "Hallo! Hoe kan ik je helpen?",
    "Estonian": "Tere! Kuidas ma saan aidata?",
    "Romanian": "Salut! Cum te pot ajuta?",
    "Persian": "سلام! چگونه می‌توانم کمک کنم؟",
    "Indonesian": "Halo! Bagaimana saya bisa membantu?",
    "Latin": "Salve! Quomodo te adiuvare possum?"
}


# --- Streamlit UI: Layout and Components ---

st.title("⚫⚪ Language Detection Bot")
st.subheader("Instantly detect and reply to messages in 22 languages.")

# Use st.container for a clean input section
with st.container(border=True):
    st.markdown("### *💬 Your Message*")
    user_input = st.text_area(
        "Enter your message below:",
        key="user_input_area",
        height=100,
        label_visibility="collapsed",
        placeholder="Type a phrase in any supported language (e.g., 'Comment ça va?')"
    )

    # Use st.columns to style the button
    col1, col2 = st.columns([1, 4])
    with col1:
        # This is the main button that triggers the logic.
        if st.button("Detect & Reply", key="detect_button", use_container_width=True):
            # The logic runs inside the button click handler below.
            pass
    # An empty column for spacing

st.markdown("---") # Visual separator

# --- Logic Execution (Triggered by the button click) ---

# Check the condition from the VISIBLE button using its key: 'detect_button'
if st.session_state.get('detect_button'):
    if user_input.strip() == "":
        st.warning("Please enter some text in the message box above.")
    else:
        # Prediction Block
        try:
            # 1. Vectorize the input
            text_vector = vectorizer.transform([user_input])

            # 2. Predict the language
            predicted_language = model.predict(text_vector)[0]

            # 3. Get the response
            reply = responses.get(
                predicted_language,
                "Sorry, I don't have a canned response for this language yet."
            )

            # Display Results in a clear, structured way
            st.success(f"✔️ Language Detected: *{predicted_language}*")

            st.markdown("### 🤖 Chatbot Reply")

            # Use st.info for the reply, which the custom CSS styles nicely
            st.info(reply)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer for a clean professional finish
st.markdown("---")
st.caption(f"App Version 1.1 | Supported Languages: 22 | Last Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
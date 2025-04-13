import streamlit as st
from utils.data_loader import load_reviews, load_faqs
from utils.faq_processor import FAQProcessor
from utils.response_generator import ResponseGenerator
import os
from dotenv import load_dotenv
import openai
import pandas as pd
from PIL import Image

# Load environment variables
load_dotenv()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'faq_processor' not in st.session_state:
    st.session_state.faq_processor = None
if 'response_gen' not in st.session_state:
    st.session_state.response_gen = None
if 'faqs_loaded' not in st.session_state:
    st.session_state.faqs_loaded = False
if 'faqs_processed' not in st.session_state:
    st.session_state.faqs_processed = False

def load_faq_database():
    """Function to load and process FAQs automatically"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found in environment variables")
        return False
    
    try:
        faqs = load_faqs()
        if not isinstance(faqs, pd.DataFrame) or faqs.empty:
            st.error("Failed to load FAQs or empty DataFrame")
            return False
        
        cache_path = os.path.join('venv', 'faq_embeddings_cache.json')
        
        st.session_state.faq_processor = FAQProcessor(
            openai_api_key,
            cache_path=cache_path
        )
        st.session_state.faq_processor.build_index(faqs)
        
        st.session_state.response_gen = ResponseGenerator(openai_api_key)
        st.session_state.faqs_processed = True
        st.session_state.faqs_loaded = True
        return True
        
    except Exception as e:
        st.error(f"Error loading FAQs: {str(e)}")
        return False

def main():
    local_css("styles.css")
    st.markdown("""
    <style>
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .title-text {
            margin-left: 20px;
            font-size: 1.8rem;
            font-weight: 600;
            color: #2d3436;
            line-height: 1.2;
        }
        img {
            background-color: transparent !important;
            mix-blend-mode: multiply;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo and subtitle
    col1, col2 = st.columns([2, 8])
    with col1:
        st.image("zaggle_logo.png", width=180)
    with col2:
        st.markdown("""
        <div class="header-container title-text">
            <h1 style="margin-left: 40px; font-size: 2rem;">AI Review Responder</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Brand Voice Selector
    brand_voice = st.selectbox(
        "Brand Voice",
        options=["Professional", "Friendly", "Supportive", "Enthusiastic"],
        index=1,
        help="Select the tone for generated responses"
    )
    
    # Automatically load FAQs if not already loaded
    if not st.session_state.get('faqs_loaded'):
        with st.spinner("🧠 Loading FAQ database..."):
            if not load_faq_database():
                st.stop()
    
    # Main content area
    # Query Input Section
    st.markdown("### ✍️ Review Input")
    user_query = st.text_area("Review text:", placeholder="Type or paste the customer review here...")
    
    # Rating Selector with visual cues
    st.markdown("### ⭐ Rating")
    rating = st.slider(
        "Select review rating (adjusts response tone):",
        1, 5, 3,
        help="1-2 stars: Empathetic | 3 stars: Neutral | 4-5 stars: Enthusiastic"
    )
    
    # Visual rating indicator
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"<span style='margin-right: 10px;'>{'★ ' if i < rating else '☆'}</span>", unsafe_allow_html=True)
    
    # Generate Response Button
    if st.button("Generate Response", key="generate_response"):
        if not user_query.strip():
            st.warning("Please enter a review first")
        else:
            with st.spinner("Analyzing review and generating response..."):
                try:
                    # FAQ Matching
                    faq_context = st.session_state.faq_processor.find_similar_faqs(
                        user_query,
                        k=3,
                        threshold=1.8
                    )
                    
                    # Response Generation
                    response = st.session_state.response_gen.generate_response(
                        user_query,
                        rating,
                        faq_context,
                        brand_voice.lower()
                    )
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("### 💬 Generated Response")
                    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
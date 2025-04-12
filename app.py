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

# Custom CSS for styling
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
    openai.api_key = openai_api_key
    client = openai
    print(client.models.list()) 
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
    # Load custom CSS
    local_css("styles.css")
    
    # Header with logo
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("zaggle_logo.png", width=80)  # Replace with your logo path
    with col2:
        st.title("Zaggle AI Review Responder")
    
    st.markdown("---")
    
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
    # Demo Mode Toggle
    demo_mode = st.checkbox("🎬 Demo Mode (Preload example queries)")
    
    # Query Input Section
    st.markdown("### ✍️ Review Input")
    if demo_mode:
        demo_options = {
            "Crisis Handling": "My card was hacked! Transactions I didn't make. How do I freeze it? Will I get my money back?",
            "Feature Discovery": "The Propel rewards are awesome! How do I redeem for flights? Any blackout dates?",
            "Multi-Question Handling": "How do I update my mobile number? Also what's the customer care email? And can I do this online?",
            "Policy Clarification": "Why was my fuel transaction declined? I have balance. What's the MCC code for fuel?",
            "Urgent Support": "My wife's medical emergency transaction failed! Need urgent help!"
        }
        selected_demo = st.selectbox("Select demo scenario:", list(demo_options.keys()))
        user_query = st.text_area("Review text:", value=demo_options[selected_demo])
    else:
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
            st.markdown(f"{'★ ' if i < rating else '☆'}")
    
    # Generate Response Button - fixed to remove the 'type' parameter
    if st.button("Generate Response", key="generate_response"):
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
                
                # FAQ References (expandable)
                if faq_context:
                    with st.expander("🔎 View FAQ References Used"):
                        for faq in faq_context:
                            st.markdown(f"**Q:** {faq['question']}")
                            st.markdown(f"**A:** {faq['answer']}")
                            st.markdown(f"*Similarity score: {faq['distance']:.2f}*")
                            st.markdown("---")
                
                # Feedback Mechanism (for demo purposes)
                st.markdown("---")
                st.markdown("### 📊 Demo Feedback")
                feedback = st.radio("How accurate was this response?", 
                                   ["Perfect", "Good", "Needs Improvement"])
                if st.button("Submit Feedback"):
                    st.success("Thanks for your feedback!")
                    
            except Exception as e:
                st.error(f"⚠️ Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
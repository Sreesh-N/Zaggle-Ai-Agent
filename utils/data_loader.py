import pandas as pd
import os

def clean_apple_reviews(file_path):
    """Clean Apple App Store reviews data"""
    df = pd.read_excel(file_path, header=None)  # No header as it's in row 3
    
    # Extract data starting from row 4 (index 3)
    apple_reviews = df.iloc[3:, [2, 3, 4, 5, 6]].copy()
    apple_reviews.columns = ['rating', 'title', 'body', 'reviewer', 'date']
    
    # Clean data
    apple_reviews['source'] = 'apple'
    apple_reviews['date'] = pd.to_datetime(apple_reviews['date'], utc=True)
    apple_reviews['text'] = apple_reviews['title'].fillna('') + ': ' + apple_reviews['body'].fillna('')
    
    return apple_reviews[['text', 'rating', 'date', 'source']]
def clean_google_reviews(file_path):
    try:
        # First try reading as Excel
        df = pd.read_excel(file_path, header=None)
        if df.shape[1] < 12:  # If not enough columns
            # Try reading as CSV
            df = pd.read_csv(file_path, header=None)
        
        print("File shape:", df.shape)
        print("First row:", df.iloc[0].tolist())
        
        # Assuming rating=column 10, text=11, date=12 (0-based 9,10,11)
        google_reviews = df.iloc[1:, [9, 10, 11]].copy()
        google_reviews.columns = ['rating', 'text', 'date']
        
        google_reviews['source'] = 'google'
        google_reviews['date'] = pd.to_datetime(google_reviews['date'], utc=True)
        return google_reviews[['text', 'rating', 'date', 'source']]
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def load_faqs():
    """Load FAQ knowledge base with proper NaN handling"""
    faq_path = "data/Chatbot FAQs.xlsx"
    if os.path.exists(faq_path):
        # Read Excel with explicit column mapping
        faqs = pd.read_excel(faq_path, usecols=["User Query", "Product Responses"])
        
        # Clean and rename columns
        faqs = faqs.rename(columns={
            "User Query": "question",
            "Product Responses": "answer"
        })
        
        # Fill NaN values with empty string
        faqs['answer'] = faqs['answer'].fillna('')
        
        # Remove completely empty rows
        faqs = faqs.dropna(how='all')
        
        return faqs[["question", "answer"]]
    return pd.DataFrame(columns=["question", "answer"])

def load_reviews(platform="all"):
    """Load and clean reviews from both sources"""
    reviews = []
    
    if platform in ["all", "apple"]:
        apple_path = "data/appstore (1).xlsx"
        if os.path.exists(apple_path):
            apple_df = clean_apple_reviews(apple_path)
            reviews.append(apple_df)
    
    if platform in ["all", "google"]:
        google_path = "data/Reviews Report 2025.xlsx"
        if os.path.exists(google_path):
            google_df = clean_google_reviews(google_path)
            if not google_df.empty:
                reviews.append(google_df)
    
    return pd.concat(reviews) if reviews else pd.DataFrame()

# Zaggle AI Review Responder

An intelligent response generator for customer reviews that combines sentiment analysis with rating-based response templates to create personalized, appropriate replies.

## Features

- **Sentiment Analysis**: Automatically detects review sentiment (positive/neutral/negative)
- **Rating-Based Responses**: Tailors responses based on star ratings (1-5)
- **FAQ Integration**: Uses existing knowledge base to provide accurate answers
- **Brand Voice Customization**: Supports multiple brand tones (Professional, Friendly, etc.)
- **Real-time Generation**: Quickly generates responses for immediate use

## Technology Stack

- **Backend**:
  - Python 3.9+
  - OpenAI GPT-4-turbo (API)
  - FAISS (for FAQ similarity search)
  - Pandas (for data handling)

- **Frontend**:
  - Streamlit (web interface)
  - Custom CSS styling

- **APIs**:
  - OpenAI API
  - (Optional) App Store/Google Play review APIs

## Setup Instructions

### Prerequisites

1. Python 3.9 or later
2. OpenAI API key
3. (Optional) FAQ dataset in CSV/Excel format

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zaggle-review-responder.git
   cd zaggle-review-responder

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a .env file in the root directory with:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
5.Prepare FAQ data:
  Place your FAQ file in data/ directory
  Ensure columns are named "question" and "answer"
  Note: The data_loader.py loads two types of files for my case so please modify the file based on your requirements

And finally to run the application ```bash streamlit run app.py ``` and you are good to go🎉🎉🎉


Happy Coding

from openai import OpenAI
import time

class ResponseGenerator:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.last_call_time = 0
        self.rate_limit_delay = 1.5

    def _analyze_sentiment(self, text):
        """Analyze sentiment of the review text using GPT-4-turbo"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "Analyze the sentiment of this review text. Respond ONLY with one word: 'positive', 'neutral', or 'negative'."
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                temperature=0.0,
                max_tokens=10
            )
            sentiment = response.choices[0].message.content.lower().strip()
            return sentiment if sentiment in ['positive', 'neutral', 'negative'] else 'neutral'
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return 'neutral'

    def _get_response_rules(self, rating, sentiment):
        """Determine response rules based on both rating and sentiment"""
        # Base rules by rating
        rules = {
            5: {
                "opening": "We're thrilled to hear about your experience!",
                "closing": "We appreciate you being a valued Zaggle customer!",
                "emoji": "🌟",
                "style": "enthusiastic and appreciative"
            },
            4: {
                "opening": "Thank you for your positive feedback!",
                "closing": "We're glad you had a good experience with Zaggle.",
                "emoji": "",
                "style": "warm and professional"
            },
            3: {
                "opening": "Thanks for sharing your feedback with us.",
                "closing": "Let us know if there's anything else we can assist with.",
                "emoji": "",
                "style": "neutral and helpful"
            },
            2: {
                "opening": "We appreciate you bringing this to our attention.",
                "closing": "Please don't hesitate to reach out if you need further assistance.",
                "emoji": "",
                "style": "solution-focused"
            },
            1: {
                "opening": "We sincerely apologize for your experience.",
                "closing": "Our support team is ready to help resolve this for you.",
                "emoji": "",
                "style": "empathetic and action-oriented"
            }
        }
        
        base_rule = rules.get(rating, rules[3])
        
        # Adjust based on sentiment
        if sentiment == "negative":
            if rating >= 4:  # High rating but negative sentiment
                base_rule["opening"] = "We appreciate your honest feedback."
                base_rule["style"] = "empathetic and solution-focused"
            base_rule["closing"] = "We're committed to improving your experience."
            base_rule["emoji"] = ""
        
        elif sentiment == "positive" and rating <= 2:
            base_rule["opening"] = "We appreciate your kind words and take your feedback seriously."
            base_rule["style"] = "appreciative and solution-focused"
            
        return base_rule

    def _format_faq_context(self, faq_context):
        """Clean FAQ formatting"""
        if not faq_context:
            return "None"
        return "\n".join([
            f"{i}. {faq['question']}\n   → {faq['answer']}" 
            for i, faq in enumerate(faq_context[:3], 1)
        ])

    def generate_response(self, review_text, review_rating, faq_context=None, brand_voice="professional"):
        """Generate response considering both rating and sentiment"""
        try:
            time_since_last = time.time() - self.last_call_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(review_text)
            
            # Get response rules
            response_rules = self._get_response_rules(review_rating, sentiment)
            
            # Build prompt
            prompt = self._build_review_response_prompt(review_text, review_rating, sentiment, faq_context, brand_voice, response_rules)
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.65,
                max_tokens=400,
                top_p=0.9
            )
            
            # Post-process and return
            return self._format_review_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            return self._fallback_response(review_text, review_rating, faq_context)

    def _build_review_response_prompt(self, review_text, rating, sentiment, faq_context, brand_voice, response_rules):
        """Build prompt specifically for review responses"""
        return f"""You are crafting an official Zaggle response to a customer review. 

    REVIEW DETAILS:
    Rating: {rating} stars
    Sentiment: {sentiment}
    Content: {review_text}

    RESPONSE GUIDELINES:
    1. Strict Formatting Rules:
    - Respond directly to the review (no greetings or closings)
    - Exactly 3 paragraphs separated by blank lines
    - Each paragraph 2-4 sentences maximum
    - Never use bullet points or lists
    - Never include contact information unless specifically about support

    2. Content Structure:
    - First paragraph: Acknowledge the feedback
    - Second paragraph: Address the main issue
    - Third paragraph: Provide resolution/next steps

    3. Tone Requirements:
    - Must sound like a natural review response
    - Avoid corporate jargon
    - Match the {sentiment} sentiment appropriately

    PROHIBITED FORMATTING:
    - Any bullet points or numbered lists
    - Email signatures or contact information
    - Greetings like "Dear customer"
    - Closings like "Best regards"
    - Paragraphs longer than 4 sentences"""

    def _format_review_response(self, response):
        """Format the response specifically for review replies"""
        # Split into paragraphs while preserving blank lines
        paragraphs = []
        current_para = []
        
        for line in response.split('\n'):
            stripped_line = line.strip()
            if not stripped_line:
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            else:
                # Skip any unwanted lines
                if not any(stripped_line.lower().startswith(word) 
                        for word in ['dear', 'hi ', 'hello', 'best', 'regards', 'sincerely']):
                    current_para.append(stripped_line)
        
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Ensure exactly 3 paragraphs with proper spacing
        if len(paragraphs) > 3:
            paragraphs = paragraphs[:3]
        elif len(paragraphs) < 3:
            paragraphs.extend([""] * (3 - len(paragraphs)))
        
        return '\n\n'.join(paragraphs)

    def _fallback_response(self, review_text, rating, faq_context):
        """Minimal fallback that still helps"""
        base = {
            5: "Thank you! We're experiencing high volume but will respond soon.",
            4: "Thanks for your patience. Our team is reviewing your query.",
            3: "We've noted your feedback.",
            2: "We're working to resolve this.",
            1: "We're prioritizing this issue."
        }.get(rating, "Thanks for your feedback.")
        
        if faq_context:
            solutions = "\n".join(f"- {faq['answer']}" for faq in faq_context[:2])
            return f"{base}\n\nTry these solutions:\n{solutions}"
        return base
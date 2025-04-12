from openai import OpenAI
import time

class ResponseGenerator:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.last_call_time = 0
        self.rate_limit_delay = 1.5  # Slightly faster than before

    def _get_rating_specific_rules(self, rating):
        """Strict rules for each rating level"""
        rules = {
            5: [
                "Use enthusiastic language ('We're thrilled!')",
                "Highlight 2 premium features",
                "Suggest one combo usage tip",
                "Include one celebratory emoji (🌟, 😊, ✨)",
                "Never offer discounts/compensation"
            ],
            4: [
                "Express gratitude ('Thank you!')",
                "Mention one premium feature",
                "Keep tone warm but professional",
                "No emojis unless user used them first"
            ],
            3: [
                "Neutral, factual tone",
                "Direct answers only",
                "No suggestions or extras"
            ],
            2: [
                "Show concern ('We'll help fix this')",
                "Provide numbered steps",
                "Offer support contact only as last resort",
                "No apologies beyond first sentence"
            ],
            1: [
                "One sincere apology maximum",
                "Immediate actionable solutions only",
                "Support contact must be specific (include phone/email)",
                "Never promise manager callbacks",
                "No compensation offers"
            ]
        }
        return "\n- ".join(rules.get(rating, rules[3]))

    def _format_faq_context(self, faq_context):
        """Clean FAQ formatting"""
        if not faq_context:
            return "None"
        return "\n".join([
            f"{i}. {faq['question']}\n   → {faq['answer']}" 
            for i, faq in enumerate(faq_context[:3], 1)
        ])

    def generate_response(self, review_text, review_rating, faq_context=None, brand_voice="professional yet approachable"):
        """Main response generation with strict controls"""
        try:
            time_since_last = time.time() - self.last_call_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            prompt = self._build_prompt(review_text, review_rating, faq_context, brand_voice)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.65,
                max_tokens=350,
                top_p=0.9
            )
            self.last_call_time = time.time()
            return self._post_process(response.choices[0].message.content)
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            return self._fallback_response(review_text, review_rating, faq_context)

    def _build_prompt(self, review_text, rating, faq_context, brand_voice):
        """Structured prompt with hard constraints"""
        # ... (keep existing tone_map code)
        
        return f"""You are Zaggle's AI support assistant. Respond to this {rating}-star review:

    REVIEW:
    {review_text}

    FAQ CONTEXT:
    {self._format_faq_context(faq_context)}

    STRICT REQUIREMENTS:
    1. ONLY suggest features that exist in these FAQs
    2. For login issues, ONLY suggest:
    - Checking internet connection
    - Updating the app
    - Resending OTP
    3. NEVER suggest these non-existent features:
    - "Remember this device"
    - Biometric login
    - Face ID authentication
    4. Format MUST use:
    - Numbered steps for solutions
    - Bullet points for explanations
    - Clear section spacing

    PROHIBITED CONTENT:
    - Any mention of unavailable features
    - Generic "try again later" advice
    - Vague suggestions without concrete steps"""

    def _post_process(self, response):
        """Clean up the AI output with strict formatting"""
        # First remove any incorrect feature suggestions
        incorrect_features = [
            "remember this device",
            "enable biometric login",
            "use face id"
        ]
        
        lines = response.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove lines suggesting non-existent features
            if any(feature in line.lower() for feature in incorrect_features):
                continue
                
            # Remove redundant apologies
            if "apologize" in line.lower() and any(l.startswith("We") for l in cleaned):
                continue
                
            # Convert paragraphs to bullet points where appropriate
            if line.startswith(("- ", "* ", "• ")):
                cleaned.append(line)
            elif len(line.split()) > 15 and not line.startswith(("1.", "2.", "3.")):
                # Split long lines into bullet points
                sentences = [s.strip() for s in line.split('. ') if s.strip()]
                for s in sentences:
                    if s:  # Skip empty sentences
                        cleaned.append(f"- {s}" if not s.endswith('.') else f"- {s}")
            else:
                cleaned.append(line)
        
        # Ensure proper spacing between sections
        formatted_response = []
        for i, line in enumerate(cleaned):
            formatted_response.append(line)
            # Add spacing after headings or before new sections
            if (line.startswith(("1.", "2.", "3.")) or 
                (i < len(cleaned)-1 and cleaned[i+1].startswith(("- ", "* ", "• ")))):
                formatted_response.append("")
        
        return '\n'.join(formatted_response)

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
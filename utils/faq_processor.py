from openai import OpenAI
import numpy as np
import faiss
import pandas as pd
import hashlib
import json
import os
from tqdm import tqdm
import time
import re
class FAQProcessor:
    def __init__(self, openai_api_key, cache_path="faq_embeddings_cache.json"):
        self.client = OpenAI(api_key=openai_api_key)
        self.index = None
        self.faq_data = None
        self.cache_file = cache_path
        self.embedding_cache = {}
        print(f"Initializing with cache path: {self.cache_file}")
        self._ensure_cache_directory()
        self._load_cache()
        print(f"Loaded {len(self.embedding_cache)} cached embeddings")
        self.rate_limit_delay = 0.2  # 200ms between API calls
        self.last_api_call = 0
        self.timeout = 10  # seconds for API timeout

    def _ensure_cache_directory(self):
        """Ensure the cache directory exists"""
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _load_cache(self):
        """Load embeddings cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                print(f"Loaded cache with {len(self.embedding_cache)} embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.embedding_cache = {}

    def _save_cache(self):
        """Ensure cache is properly saved"""
        try:
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Write to temporary file first
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.embedding_cache, f)
            
            # Atomic rename
            os.replace(temp_file, self.cache_file)
            print(f"Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"Cache save failed: {str(e)}")
            raise

    def _get_cache_key(self, text):
        """Generate consistent cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def embed_batch(self, texts):
        """Process multiple texts in a single API call"""
        # Filter out cached texts
        to_process = []
        cache_keys = []
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            cache_key = self._get_cache_key(text)
            if cache_key not in self.embedding_cache:
                to_process.append(text)
                cache_keys.append(cache_key)
        
        if not to_process:
            return
        
        # Rate limiting
        time_since_last = time.time() - self.last_api_call
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            response = self.client.embeddings.create(
                input=to_process,
                model="text-embedding-3-small",
                timeout=self.timeout
            )
            self.last_api_call = time.time()
            
            # Store all results
            for i, embedding in enumerate(response.data):
                self.embedding_cache[cache_keys[i]] = embedding.embedding
            
            self._save_cache()
        except Exception as e:
            print(f"Batch embedding error: {e}")

    def embed_text(self, text):
        """Get embedding from cache or API with faster rate limiting"""
        if not text or not isinstance(text, str):
            return None
            
        cache_key = self._get_cache_key(text)
        
        # Return cached embedding if available
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        # Faster rate limiting (50ms instead of 100ms)
        time_since_last = time.time() - self.last_api_call
        if time_since_last < 0.05:  # 50ms
            time.sleep(0.05 - time_since_last)
        
        # Get embedding from API with timeout
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
                timeout=10  # seconds
            )
            self.last_api_call = time.time()
            embedding = response.data[0].embedding
            self.embedding_cache[cache_key] = embedding
            self._save_cache()
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    # def build_index(self, faqs_df):
    #     """Build FAISS index with batch processing"""
    #     if faqs_df is None or len(faqs_df) == 0:
    #         raise ValueError("No FAQs provided")
            
    #     self.faq_data = faqs_df
    #     embeddings = []
        
    #     print(f"Processing {len(faqs_df)} FAQs ({len(self.embedding_cache)} cached)")
        
    #     # First pass - try to get all from cache
    #     cached_count = 0
    #     for _, row in tqdm(faqs_df.iterrows(), total=len(faqs_df), desc="Checking cache"):
    #         combined_text = f"{row['question']} {row['answer']}"
    #         embedding = self.embedding_cache.get(self._get_cache_key(combined_text))
    #         if embedding:
    #             embeddings.append(embedding)
    #             cached_count += 1
        
    #     print(f"Found {cached_count} cached embeddings")
        
    #     # Second pass - only process uncached items
    #     if cached_count < len(faqs_df):
    #         print(f"Processing {len(faqs_df)-cached_count} new embeddings")
    #         for _, row in tqdm(faqs_df.iterrows(), total=len(faqs_df), desc="Generating embeddings"):
    #             combined_text = f"{row['question']} {row['answer']}"
    #             if self._get_cache_key(combined_text) not in self.embedding_cache:
    #                 embedding = self.embed_text(combined_text)
    #                 if embedding:
    #                     embeddings.append(embedding)
        
    #     if not embeddings:
    #         raise ValueError("No valid embeddings generated")
            
    #     embeddings = np.array(embeddings).astype('float32')
    #     self.index = faiss.IndexFlatL2(embeddings.shape[1])
    #     self.index.add(embeddings)
    #     print(f"Built index with {len(embeddings)} embeddings")
    def build_index(self, faqs_df, batch_size=50, progress_callback=None):
        """Build index with batch processing and progress tracking"""
        if faqs_df is None or len(faqs_df) == 0:
            raise ValueError("No FAQs provided")
        
        self.faq_data = faqs_df
        embeddings = []
        total_items = len(faqs_df)
        processed_count = 0
        
        # First get all cached embeddings
        cached_count = 0
        texts_to_process = []
        
        for _, row in faqs_df.iterrows():
            combined_text = f"{row['question']} {row['answer']}"
            cache_key = self._get_cache_key(combined_text)
            
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                cached_count += 1
            else:
                texts_to_process.append(combined_text)
            
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_items)
        
        print(f"Found {cached_count} cached embeddings")
        print(f"Processing {len(texts_to_process)} new embeddings in batches")
        
        # Process remaining in batches
        for i in range(0, len(texts_to_process), batch_size):
            batch = texts_to_process[i:i+batch_size]
            self.embed_batch(batch)
            
            # Add newly processed embeddings
            for text in batch:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
            
            processed_count += len(batch)
            if progress_callback:
                progress_callback(min(processed_count, total_items), total_items)
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Built index with {len(embeddings)} embeddings")
        
        if progress_callback:
            progress_callback(total_items, total_items)

    # In faq_processor.py
    def find_similar_faqs(self, query_text, k=5, threshold=1.5):
        """Find answers for all questions in multi-question queries"""
        # First detect if there are multiple questions
        questions = self._split_questions(query_text)
        
        results = []
        for q in questions:
            query_embedding = self.embed_text(q.lower().strip())
            if query_embedding is None:
                continue
                
            query_embedding = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_embedding, k)
            
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.faq_data):
                    continue
                    
                faq = self.faq_data.iloc[idx]
                if not faq['answer'] or pd.isna(faq['answer']):
                    continue
                    
                if distances[0][i] <= threshold:
                    results.append({
                        "question": q,  # Store the original sub-question
                    "matched_faq": faq['question'],
                    "answer": str(faq['answer']).strip(),
                    "distance": float(distances[0][i])
                })
    
        # Deduplicate while keeping best matches
        seen_questions = set()
        final_results = []
        for r in sorted(results, key=lambda x: x['distance']):
            if r['question'] not in seen_questions:
                final_results.append(r)
                seen_questions.add(r['question'])
        return final_results

    def _split_questions(self, text):
        """Split compound questions into individual questions"""
        # Simple heuristic - split on question marks or "and"/"also"
        questions = []
        for part in re.split(r'\?|\band\b|\balso\b', text):
            part = part.strip()
            if part and any(c.isalpha() for c in part):
                # Add question mark back if needed
                if not part.endswith('?'):
                    part += '?'
                questions.append(part)
        return questions if questions else [text]
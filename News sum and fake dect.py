import requests
import re
import random
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from newspaper import Article
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import numpy as np
from datetime import datetime
import openai
import concurrent.futures
import os
from sklearn.datasets import fetch_20newsgroups  # Moved import here

# Initialize NLTK (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Configuration - replace these with your actual API keys or set as environment variables
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_newsapi_key')  # Get from https://newsapi.org/
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_key')  # For advanced summarization
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_API_KEY', 'your_perspective_key')  # For toxicity analysis (optional)

class NewsScraper:
    def __init__(self):
        self.sources = [
            'https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey=' + NEWS_API_KEY,
            'https://newsapi.org/v2/top-headlines?sources=cnn&apiKey=' + NEWS_API_KEY,
            'https://newsapi.org/v2/top-headlines?sources=reuters&apiKey=' + NEWS_API_KEY
        ]
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'

    def scrape_news_api(self):
        articles = []
        for source in self.sources:
            try:
                response = requests.get(source, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                if data.get('status') == 'ok':
                    articles.extend(data.get('articles', []))
            except Exception as e:
                print(f"Error scraping {source}: {str(e)}")
        return articles

    def scrape_custom_sources(self, urls):
        articles = []
        for url in urls:
            try:
                url = url.strip()  # Clean the URL
                article = Article(url, headers={'User-Agent': self.user_agent})
                article.download()
                article.parse()

                # Handle cases where metadata might be missing
                article_data = {
                    'title': article.title or 'No title available',
                    'text': article.text or '',
                    'url': url,
                    'source': article.source_url or 'Unknown source',
                    'authors': article.authors or ['Unknown author'],
                    'publish_date': article.publish_date or datetime.now()
                }
                articles.append(article_data)
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
        return articles

    def get_news(self, use_api=True, custom_urls=None):
        if use_api:
            return self.scrape_news_api()
        elif custom_urls:
            # Filter out empty URLs
            valid_urls = [url for url in custom_urls if url.strip()]
            if valid_urls:
                return self.scrape_custom_sources(valid_urls)
        return []

class NewsSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        try:
            # Load pre-trained summarization model with error handling
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error loading summarization model: {str(e)}")
            self.summarizer = None

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s.,;:!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extractive_summarize(self, text, num_sentences=3):
        if not text:
            return ""

        try:
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            if not sentences:
                return ""

            # Tokenize words and create frequency table
            words = word_tokenize(text.lower())
            freq_table = {}

            for word in words:
                if word not in self.stop_words and word.isalpha():  # Only count alphabetic words
                    freq_table[word] = freq_table.get(word, 0) + 1

            # Score sentences based on word frequency
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                for word in word_tokenize(sentence.lower()):
                    if word in freq_table:
                        sentence_scores[i] = sentence_scores.get(i, 0) + freq_table[word]

            # Get top N sentences
            num_sentences = min(num_sentences, len(sentences))
            if num_sentences <= 0:
                return ""

            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            top_sentences = sorted([i for i, _ in top_sentences])

            # Construct the summary
            summary = ' '.join([sentences[i] for i in top_sentences])
            return summary
        except Exception as e:
            print(f"Error in extractive summarization: {str(e)}")
            return text[:200] + "..."  # Fallback to first 200 characters

    def abstractive_summarize(self, text, max_length=130, min_length=30):
        if not text or not self.summarizer:
            return ""

        try:
            text = self.preprocess_text(text)
            if len(text.split()) < 10:  # Don't summarize very short texts
                return text

            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error in abstractive summarization: {str(e)}")
            return self.extractive_summarize(text)

    def gpt_summarize(self, text):
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_key':
            print("OpenAI API key not configured")
            return self.abstractive_summarize(text)

        openai.api_key = OPENAI_API_KEY
        try:
            if not text or len(text.split()) < 10:
                return text

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes news articles concisely."},
                    {"role": "user", "content": f"Summarize this article in 3-5 sentences:\n\n{text[:3000]}"}  # Limit input size
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error with GPT summarization: {str(e)}")
            return self.abstractive_summarize(text)

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        try:
            # Try to load pre-trained model
            if os.path.exists('fake_news_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                with open('fake_news_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
        except Exception as e:
            print(f"Error loading fake news model: {str(e)}")

        # Fallback to a simple model if loading failed
        if not self.model or not self.vectorizer:
            print("Using fallback fake news detection model")
            try:
                # Using a smaller dataset for the fallback model
                from sklearn.datasets import fetch_20newsgroups
                categories = ['talk.politics.misc', 'sci.space']  # Just use 2 categories for demo
                newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

                self.vectorizer = TfidfVectorizer(max_features=1000)
                X_train = self.vectorizer.fit_transform(newsgroups_train.data)
                self.model = PassiveAggressiveClassifier(max_iter=50)
                self.model.fit(X_train, newsgroups_train.target)
            except Exception as e:
                print(f"Error creating fallback model: {str(e)}")
                self.model = None
                self.vectorizer = None

    def predict_fake_news(self, text):
        if not text or not self.model or not self.vectorizer:
            return {
                'is_fake': False,
                'confidence': 0.5,
                'explanation': "Model not available for fake news detection"
            }

        try:
            # Vectorize the text
            text_vec = self.vectorizer.transform([text])

            # Make prediction
            proba = self.model.decision_function(text_vec)
            confidence = 1 / (1 + np.exp(-proba[0]))  # Convert to pseudo-probability

            # Use a more conservative threshold
            is_fake = confidence > 0.7  # Higher threshold to reduce false positives

            return {
                'is_fake': bool(is_fake),
                'confidence': float(np.clip(confidence, 0, 1)),  # Ensure between 0-1
                'explanation': self.generate_explanation(text, is_fake, confidence)
            }
        except Exception as e:
            print(f"Error in fake news prediction: {str(e)}")
            return {
                'is_fake': False,
                'confidence': 0.5,
                'explanation': "Error analyzing article reliability"
            }

    def generate_explanation(self, text, is_fake, confidence):
        if not text:
            return "No text available for analysis"

        suggestions = [
            "Check the publication date and author credentials.",
            "Look for corroborating sources on reputable news sites.",
            "Be wary of emotional language or sensational claims.",
            "Verify facts with official sources or fact-checking websites."
        ]

        if is_fake:
            return (f"This article shows signs of being unreliable (confidence: {confidence:.2f}). "
                    f"Suggestions: {' '.join(suggestions)}")
        else:
            return (f"This article appears potentially reliable (confidence: {confidence:.2f}), "
                    f"but always verify information. {random.choice(suggestions)}")

    def analyze_toxicity(self, text):
        if not PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEY == 'your_perspective_key' or not text:
            return None

        url = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'
        params = {'key': PERSPECTIVE_API_KEY}
        data = {
            'comment': {'text': text[:1000]},  # Limit text size for API
            'requestedAttributes': {
                'TOXICITY': {}, 'INSULT': {}, 'SPAM': {},
                'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}
            },
            'languages': ['en']
        }

        try:
            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()

            scores = {}
            for attr in data['requestedAttributes']:
                if attr in result.get('attributeScores', {}):
                    scores[attr.lower()] = result['attributeScores'][attr]['summaryScore']['value']

            return {
                **scores,
                'is_toxic': scores.get('toxicity', 0) > 0.7 or scores.get('severe_toxicity', 0) > 0.7
            }
        except Exception as e:
            print(f"Error with Perspective API: {str(e)}")
            return None

class NewsAnalyzerApp:
    def __init__(self):
        self.scraper = NewsScraper()
        self.summarizer = NewsSummarizer()
        self.detector = FakeNewsDetector()

    def analyze_news(self, use_api=True, custom_urls=None):
        # Get news articles with error handling
        try:
            articles = self.scraper.get_news(use_api=use_api, custom_urls=custom_urls)
            if not articles:
                print("No articles found or retrieved.")
                return []
        except Exception as e:
            print(f"Error retrieving news: {str(e)}")
            return []

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Limit workers
            futures = []
            for article in articles:
                try:
                    if use_api:
                        # For NewsAPI articles
                        content = f"{article.get('title', '')}\n\n{article.get('description', '')}\n\n{article.get('content', '')}"
                        futures.append(executor.submit(
                            self.process_article,
                            article.get('title', 'No title'),
                            content,
                            article.get('url', ''),
                            article.get('source', {}).get('name', 'Unknown')
                        ))
                    else:
                        # For custom scraped articles
                        futures.append(executor.submit(
                            self.process_article,
                            article.get('title', 'No title'),
                            article.get('text', ''),
                            article.get('url', ''),
                            article.get('source', 'Unknown')
                        ))
                except Exception as e:
                    print(f"Error preparing article for processing: {str(e)}")

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:  # Only append valid results
                        results.append(result)
                except Exception as e:
                    print(f"Error processing article: {str(e)}")

        return results

    def process_article(self, title, text, url, source):
        if not text or not isinstance(text, str):
            text = ""

        try:
            # Clean and preprocess text
            text = self.summarizer.preprocess_text(text)

            # Summarize - try abstractive first, fall back to extractive
            summary = ""
            try:
                summary = self.summarizer.abstractive_summarize(text)
                if not summary or len(summary.split()) < 5:  # If summary is too short
                    summary = self.summarizer.extractive_summarize(text)
            except Exception as e:
                print(f"Summarization error: {str(e)}")
                summary = text[:200] + "..."  # Fallback

            # Fake news detection
            fake_news_result = self.detector.predict_fake_news(text)

            # Toxicity analysis (optional)
            toxicity_result = None
            if len(text.split()) > 20:  # Only analyze longer texts
                toxicity_result = self.detector.analyze_toxicity(text)

            return {
                'title': title or 'No title',
                'source': source or 'Unknown source',
                'url': url or '',
                'summary': summary,
                'fake_news_analysis': fake_news_result,
                'toxicity_analysis': toxicity_result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error processing article: {str(e)}")
            return None

    def save_results(self, results, filename='news_analysis.csv'):
        if not results:
            print("No results to save")
            return

        try:
            df = pd.DataFrame([r for r in results if r is not None])  # Filter out None results

            # Flatten nested dictionaries with error handling
            df_fake = pd.json_normalize(df['fake_news_analysis'])
            df_fake.columns = [f'fake_{col}' for col in df_fake.columns]

            df_tox = pd.json_normalize(df['toxicity_analysis'])
            if not df_tox.empty:
                df_tox.columns = [f'tox_{col}' for col in df_tox.columns]

            df = pd.concat([
                df.drop(['fake_news_analysis', 'toxicity_analysis'], axis=1),
                df_fake,
                df_tox
            ], axis=1)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Successfully saved results to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

def main():
    app = NewsAnalyzerApp()

    print("News Analyzer")
    print("1. Analyze top news from major sources")
    print("2. Analyze custom news URLs")
    choice = input("Enter your choice (1 or 2): ").strip()

    results = []
    if choice == '1':
        print("Fetching top news...")
        results = app.analyze_news(use_api=True)
    elif choice == '2':
        urls = input("Enter URLs separated by commas: ").split(',')
        results = app.analyze_news(use_api=False, custom_urls=urls)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    if not results:
        print("No results to display. There may have been an error fetching or processing articles.")
        return

    # Display results
    for i, result in enumerate(results, 1):
        if not result:
            continue

        print(f"\n\n--- Article {i} ---")
        print(f"Title: {result.get('title', 'No title')}")
        print(f"Source: {result.get('source', 'Unknown source')}")
        print(f"URL: {result.get('url', 'No URL')}")
        print("\nSummary:")
        print(result.get('summary', 'No summary available'))

        fake_analysis = result.get('fake_news_analysis', {})
        print("\nFake News Analysis:")
        print(f" - Is Fake: {fake_analysis.get('is_fake', 'Unknown')}")
        print(f" - Confidence: {fake_analysis.get('confidence', 0):.2f}")
        print(f" - Explanation: {fake_analysis.get('explanation', 'Analysis not available')}")

        tox_analysis = result.get('toxicity_analysis')
        if tox_analysis:
            print("\nToxicity Analysis:")
            print(f" - Toxicity Score: {tox_analysis.get('toxicity', 0):.2f}")
            print(f" - Is Toxic: {tox_analysis.get('is_toxic', False)}")
            if tox_analysis.get('is_toxic', False):
                print("   Warning: This content may contain toxic elements.")

    # Save results
    try:
        app.save_results(results)
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()

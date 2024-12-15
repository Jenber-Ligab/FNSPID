# src/sentiment_analysis.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Tuple, List, Dict

# Ensure you have the required nltk resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    
    @staticmethod
    def analyze_sentiment(headlines: pd.Series) -> pd.DataFrame:
        """
        Analyze sentiment of headlines using VADER.

        Parameters:
        - headlines (pd.Series): Series of headline strings.

        Returns:
        - pd.DataFrame: DataFrame with original headlines and their sentiment scores.
        """
        sia = SentimentIntensityAnalyzer()
        sentiments = headlines.apply(lambda x: sia.polarity_scores(x))
        sentiment_df = pd.DataFrame(sentiments.tolist())
        sentiment_df = pd.concat([headlines, sentiment_df], axis=1)
        return sentiment_df

    @staticmethod
    def categorize_sentiment(compound_score: float) -> str:
        """
        Categorize sentiment based on the compound score.

        Parameters:
        - compound_score (float): The compound score from VADER.

        Returns:
        - str: 'Positive', 'Neutral', or 'Negative'.
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    @staticmethod
    def apply_sentiment_categories(data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sentiment categories to the DataFrame.

        Parameters:
        - data (pd.DataFrame): DataFrame with sentiment analysis.

        Returns:
        - pd.DataFrame: DataFrame with an additional 'Sentiment' column.
        """
        data['Sentiment'] = data['compound'].apply(SentimentAnalyzer.categorize_sentiment)
        return data

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the input text by removing punctuation, stopwords, and non-alphabetic characters.
        
        Parameters:
        - text (str): The input text to preprocess.
        
        Returns:
        - str: The cleaned and preprocessed text.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

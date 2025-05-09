
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Ensure required NLTK resources are downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Sample social media conversations (you can replace this with your own data)
conversations = [
    "I love this new movie, it's amazing! 😊",
    "This is terrible, I can't believe this is happening! 😡",
    "I feel okay about the situation, not great but not bad.",
    "Wow, that was a fantastic experience! 😍",
    "I'm feeling so down, everything is going wrong. 😔"
]

# Create a function to classify sentiment
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    # Classifying based on compound score
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Analyze each conversation and store the results
results = []
for text in conversations:
    sentiment = analyze_sentiment(text)
    results.append({'Text': text, 'Sentiment': sentiment})

# Create a DataFrame to display results
df = pd.DataFrame(results)

# Display the results
print(df)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Load the IMDB dataset
df = pd.read_csv('imdb.csv')

# Clean data
df = df.dropna()

# Convert sentiment to numeric values (positive=1, negative=0)
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

def analyze_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "positive" if prediction == 1 else "negative"

user_text = input("Enter a review to analyze: ")
sentiment = analyze_sentiment(user_text)
print(f"The sentiment is: {sentiment}")

vader = SentimentIntensityAnalyzer()
scores = vader.polarity_scores(user_text)
print(scores)

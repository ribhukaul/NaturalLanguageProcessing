

import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

# Ensure you have the stopwords dataset downloaded
nltk.download('stopwords')

# Load your reviews data from the specified path
reviews = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/amazon_reviews.csv')

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning function to your reviews
reviews['cleaned_text'] = reviews['body'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
reviews['cleaned_text'] = reviews['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Save the cleaned reviews to a CSV file
reviews[['cleaned_text']].to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/cleaned_amazon_reviews1.csv', index=False)

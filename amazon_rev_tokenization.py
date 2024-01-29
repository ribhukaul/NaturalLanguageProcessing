import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Ensure you have the punkt dataset downloaded for tokenization
nltk.download('punkt')

# Load the cleaned text data
cleaned_data = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/cleaned_amazon_reviews1.csv')

# Tokenize the text
cleaned_data['tokens'] = cleaned_data['cleaned_text'].apply(word_tokenize)

# Select only the 'tokens' column
tokens_data = cleaned_data[['tokens']]

# Save the 'tokens' data to a new CSV file
tokens_data.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/tokenized_amazon_reviews.csv', index=False)

# Optional: Print the first few rows to verify
print(tokens_data.head())

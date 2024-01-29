from transformers import pipeline
import pandas as pd
import ast
# Load the sentiment-analysis pipeline
nlp = pipeline("sentiment-analysis")
# Define a function to apply BERT sentiment analysis
def bert_sentiment(tokens):
    # Join tokens into a single string
    text = " ".join(tokens)
    # Get the sentiment label directly using the pipeline
    return nlp(text, truncation=True, max_length=512)[0]['label']
# Load the dataset
cleaned_reviews_df = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/tokenized_amazon_reviews.csv')
# Convert string representation of list to actual list
cleaned_reviews_df['tokens'] = cleaned_reviews_df['tokens'].apply(ast.literal_eval)
# Apply the function to get sentiment labels
cleaned_reviews_df['bert_sentiment'] = cleaned_reviews_df['tokens'].head(100).apply(bert_sentiment)
# Save the results
cleaned_reviews_df.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/bert_sentiment_reviews.csv', index=False)



# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Set the aesthetic style of the plots
sns.set_style("whitegrid")
# Create the histogram
plt.figure(figsize=(10, 6))
sns.countplot(x='bert_sentiment', data=cleaned_reviews_df,
              order=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
              palette={'POSITIVE': 'green', 'NEUTRAL': 'blue', 'NEGATIVE': 'red'})
# Add title and labels
plt.title('Distribution of Sentiments - BERT Model')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
# Show the plot
plt.show()



# emotion analysis

import pandas as pd
from transformers import pipeline

# Load the cleaned reviews
cleaned_reviews_df = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/cleaned_amazon_reviews1.csv')

# Load the sentiment-analysis pipeline from Hugging Face Transformers
nlp_sentiment = pipeline("sentiment-analysis")

# Load the emotion-analysis pipeline from Hugging Face Transformers
# You might need to replace 'bhadresh-savani/distilbert-base-uncased-emotion' with the correct model name
nlp_emotion = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Define a function to apply BERT sentiment analysis
def bert_sentiment(text):
    return nlp_sentiment(text)[0]['label']

# Define a function to apply BERT emotion analysis
def bert_emotion(text):
    return nlp_emotion(text)[0]['label']

# Apply the functions to get sentiment and emotion labels
# Note: BERT is resource-intensive, consider applying it to a subset of data first
cleaned_reviews_df['bert_sentiment'] = cleaned_reviews_df['cleaned_text'].head(100).apply(bert_sentiment)
cleaned_reviews_df['bert_emotion'] = cleaned_reviews_df['cleaned_text'].head(100).apply(bert_emotion)

# Save the results with sentiment and emotion labels
cleaned_reviews_df.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/bert_sentiment_emotion_reviews.csv', index=False)

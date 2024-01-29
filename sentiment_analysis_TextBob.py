from textblob import TextBlob
import pandas as pd

# Load the cleaned reviews
cleaned_reviews_df = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/cleaned_amazon_reviews1.csv')

# Define a function to apply TextBlob sentiment analysis
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to get sentiment scores
cleaned_reviews_df['textblob_sentiment'] = cleaned_reviews_df['cleaned_text'].apply(textblob_sentiment)

# Save the results
cleaned_reviews_df.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/textblob_sentiment_reviews.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data with TextBlob sentiment scores
textblob_sentiments_df = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/textblob_sentiment_reviews.csv')

# Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.histplot(textblob_sentiments_df['textblob_sentiment'], kde=True, bins=30)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Categorization of Sentiments
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

textblob_sentiments_df['sentiment_category'] = textblob_sentiments_df['textblob_sentiment'].apply(categorize_sentiment)

# Visualization of Sentiment Categories with specific colors
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_category', 
              data=textblob_sentiments_df, 
              order=['Positive', 'Neutral', 'Negative'],
              palette={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'})
plt.title('Count of Reviews by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.show()

# Save the updated DataFrame with sentiment categories
textblob_sentiments_df.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/categorized_sentiment_reviews.csv', index=False)

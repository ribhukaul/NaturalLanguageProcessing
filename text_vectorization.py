import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the tokenized data
tokenized_data = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/tokenized_amazon_reviews.csv')

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Apply TF-IDF to the tokenized texts (join tokens into a single string)
tf_idf_matrix = vectorizer.fit_transform(tokenized_data['tokens'].apply(lambda x: ' '.join(eval(x))))

# Convert the TF-IDF matrix to a DataFrame (for better visualization and further processing)
tf_idf_df = pd.DataFrame(tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save the TF-IDF DataFrame to a new CSV file
tf_idf_df.to_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/vectorized_amazon_reviews.csv', index=False)

# Optional: Print the first few rows to verify
print(tf_idf_df.head())

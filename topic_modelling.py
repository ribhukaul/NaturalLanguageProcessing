import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import ast
# Function to convert string representation of list to actual list
def convert_to_list(string):
    return ast.literal_eval(string)
# Load the tokenized reviews
tokenized_reviews_path = 'D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/tokenized_amazon_reviews.csv'
tokenized_reviews_df = pd.read_csv(tokenized_reviews_path)
# Apply the conversion function to the 'tokens' column
tokenized_reviews_df['tokens'] = tokenized_reviews_df['tokens'].apply(convert_to_list)
# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(tokenized_reviews_df['tokens'])
# Filter out extremes to remove tokens that appear too frequently or too rarely
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
# Convert document into the bag-of-words (BoW) format
corpus = [dictionary.doc2bow(text) for text in tokenized_reviews_df['tokens']]
# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)
# Save the model to disk
lda_model.save('lda_model.gensim')
# Save the dictionary and corpus to disk for future use
dictionary.save('dictionary.gensim')
corpora.MmCorpus.serialize('corpus.mm', corpus)
# Retrieve the 10 topics from the LDA model
topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
# Create lists to store topic labels and words
topic_labels = []
topic_words = []
# Iterate through each topic and add information to the lists
for topic_num, topic_words_list in topics:
    topic_label = f"Topic {topic_num}"
    words = ", ".join([word for word, _ in topic_words_list])
    topic_labels.append(topic_label)
    topic_words.append(words)
# Create a DataFrame from the lists
topics_df = pd.DataFrame({'Topic': topic_labels, 'Words': topic_words})
# Display the DataFrame
print(topics_df)

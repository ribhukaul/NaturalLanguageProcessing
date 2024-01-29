import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load your dataset
df = pd.read_csv('D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/amazon_reviews.csv')
texts = df['body'].tolist()

# Tokenize and encode sentences in batches
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform mean pooling to get sentence embeddings
embeddings = model_output.last_hidden_state.mean(dim=1).numpy()

# Perform KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(embeddings)

# Assign clusters to each example
df['cluster'] = kmeans.labels_

# Reduce the dimensionality of embeddings to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
df['pca1'] = reduced_embeddings[:, 0]
df['pca2'] = reduced_embeddings[:, 1]

# Plotting the clusters with PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette=sns.color_palette("hsv", num_clusters))
plt.title('BERT Embeddings Clustered with KMeans')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster', loc='best')
plt.show()

# Manually determined themes for each cluster
cluster_themes = {
    0: 'Customer Service',
    1: 'Product Quality',
    2: 'Shipping and Delivery',
    3: 'Price Satisfaction',
    4: 'User Experience'
}

# Map the cluster numbers to theme names
df['theme'] = df['cluster'].map(cluster_themes)

# Plot the distribution of themes
plt.figure(figsize=(10, 6))
sns.countplot(x='theme', data=df, palette='Set3')
plt.title('Distribution of Themes Across Clusters')
plt.xlabel('Theme')
plt.xticks(rotation=45)
plt.ylabel('Number of Reviews')
plt.show()

import pandas as pd
import networkx as nx
from itertools import combinations
import ast
import matplotlib.pyplot as plt

# Load the tokenized data
file_path = "D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem3/S3-WebnSocialMining/ExamSubmission/tokenized_amazon_reviews.csv"
tokenized_data_df = pd.read_csv(file_path)

# Convert the string representations of lists back into actual lists
tokenized_data_df['tokens'] = tokenized_data_df['tokens']. apply(ast.literal_eval)

# Create a new graph
G = nx.Graph()

# Iterate over the tokenized reviews and add edges between all tokens within a review
for token_list in tokenized_data_df['tokens']:
    for token_pair in combinations(token_list, 2):
        if G.has_edge(*token_pair):
            G[token_pair[0]][token_pair[1]]['weight'] += 1
        else:
            G.add_edge(token_pair[0], token_pair[1], weight=1)

# Set a threshold value to reduce the number of edges, but low enough to keep the graph informative
threshold = 15  # Adjust this value based on your dataset

# Create a new graph with thresholding
G_thresholded = nx.Graph()
for (u, v, d) in G.edges(data=True):
    if d['weight'] >= threshold:
        G_thresholded.add_edge(u, v, weight=d['weight'])

# Calculate the degree of each node in the thresholded graph
degrees = dict(G_thresholded.degree())

# Visualization setup with figure and axis
fig, ax = plt.subplots(figsize=(20, 20))  # Increase figure size

# Use a Kamada-Kawai layout for the positions of the nodes
pos = nx.kamada_kawai_layout(G_thresholded)

# Draw the network graph
nodes = nx.draw_networkx_nodes(
    G_thresholded, pos, ax=ax,
    node_size=[v * 100 for v in degrees.values()],  # Same node size as before
    node_color=list(degrees.values()),
    cmap=plt.cm.viridis,
    alpha=0.9
)

# Draw edges with a reduced width to make the stems thinner
edge_weights = [d['weight'] * 0.01 for (u, v, d) in G_thresholded.edges(data=True)]  # Reduce edge width here
edges = nx.draw_networkx_edges(G_thresholded, pos, ax=ax, width=edge_weights, alpha=0.4)

# Label a larger number of nodes to show more words
label_threshold = 10  # Nodes with a degree higher than this will be labeled
labels = {node: node for node, degree in degrees.items() if degree > label_threshold}
nx.draw_networkx_labels(G_thresholded, pos, labels=labels, ax=ax, font_size=10)

# Set the plot title
ax.set_title("Word Co-occurrence Network (Adjusted Node Sizes and Thinner Edges)", fontsize=20)

# Turn off the axis for a cleaner look
plt.axis('off')

# Show the plot
plt.show()

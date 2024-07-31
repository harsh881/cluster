import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the keywords
def load_keywords(filename):
    return pd.read_csv(filename)

# Generate embeddings using Sentence Transformers
def generate_embeddings(keywords):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(keywords['Keyword'], show_progress_bar=True)
    return embeddings

# Cluster the data
def cluster_data(embeddings, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

# Visualize the silhouette scores to determine the optimal number of clusters
def visualize_silhouette_scores(embeddings, max_clusters=15):
    from sklearn.metrics import silhouette_samples, silhouette_score
    import numpy as np

    silhouette_scores = []
    range_n_clusters = list(range(2, max_clusters + 1))

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(range_n_clusters, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Various Cluster Counts')
    plt.show()

# Determine the cluster names by picking a keyword from each cluster
def name_clusters(data, clusters):
    data['Cluster'] = clusters
    cluster_names = {}
    for cluster_id in sorted(data['Cluster'].unique()):
        cluster_keywords = data[data['Cluster'] == cluster_id]['Keyword']
        representative_keyword = cluster_keywords.iloc[0]  # Simplistic approach
        cluster_names[cluster_id] = representative_keyword
    return cluster_names

# Save the clustered keywords
def save_clustered_keywords(data, filename='clustered_keywords.csv'):
    data.to_csv(filename, index=False)

def main():
    filename = 'keywords.csv'
    keywords = load_keywords(filename)
    embeddings = generate_embeddings(keywords)
    clusters = cluster_data(embeddings, num_clusters=10)
    cluster_names = name_clusters(keywords, clusters)
    keywords['Cluster Name'] = keywords['Cluster'].map(cluster_names)
    save_clustered_keywords(keywords)

if __name__ == "__main__":
    main()

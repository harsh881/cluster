import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Load the keywords
def load_keywords(filename):
    return pd.read_csv(filename)

# Preprocess keywords: clean and normalize
def preprocess_keywords(data):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    data['Keyword'] = data['Keyword'].str.lower().str.replace('[^\w\s]', '').str.strip()
    data['Keyword'] = data['Keyword'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return data

# Create TF-IDF Vectors
def vectorize_keywords(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Keyword'])
    return X, vectorizer

# Cluster the data
def cluster_data(X, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

# Generate descriptive cluster names
def name_clusters(data, clusters, vectorizer, num_clusters=10):
    data['Cluster'] = clusters
    cluster_names = {}
    for i in range(num_clusters):
        cluster_keywords = data[data['Cluster'] == i]['Keyword']
        all_words = ' '.join(cluster_keywords).split()
        most_common = Counter(all_words).most_common(3)
        cluster_name = ' '.join(word for word, freq in most_common)
        cluster_names[i] = cluster_name if cluster_name else f'Cluster {i}'
    return cluster_names

# Save the clustered keywords
def save_clustered_keywords(data, cluster_names, filename='clustered_keywords.csv'):
    data['Cluster Name'] = data['Cluster'].apply(lambda x: cluster_names[x])
    data = data.sort_values(by=['Cluster Name'])
    data[['Cluster Name', 'Keyword', 'Volume']].to_csv(filename, index=False)

def main():
    filename = 'keywords.csv'
    num_clusters = 10  # Adjust based on your needs
    keywords = load_keywords(filename)
    keywords = preprocess_keywords(keywords)
    X, vectorizer = vectorize_keywords(keywords)
    
    clusters, kmeans = cluster_data(X, num_clusters=num_clusters)
    cluster_names = name_clusters(keywords, clusters, vectorizer, num_clusters=num_clusters)
    save_clustered_keywords(keywords, cluster_names)
    print("Keywords are clustered")
if __name__ == "__main__":
    main()

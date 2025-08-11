from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Select categories
categories = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.mideast']
newsgroups = fetch_20newsgroups(
    subset='all', 
    categories=categories, 
    remove=('headers', 'footers', 'quotes')
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=5000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# K-Means clustering
true_k = 4
kmeans = KMeans(n_clusters=true_k, random_state=42)
kmeans.fit(X_tfidf)

# Top terms in each cluster
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("Top terms per cluster:")
for i in range(true_k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_terms)}")

# Reduce dimensions for visualization
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap='tab10', s=10)
plt.title("K-Means Clustering of News Articles")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.show()

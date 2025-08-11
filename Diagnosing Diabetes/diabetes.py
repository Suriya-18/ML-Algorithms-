import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ Load dataset
df = pd.read_csv("diabetes.csv")

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ✅ Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train SVM
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)

# ✅ Predictions
y_pred = model.predict(X_test)

# ✅ Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Retrain model on PCA-reduced data for decision boundary
model_pca = SVC(kernel='rbf', C=1, gamma='scale')
model_pca.fit(X_pca, y)

# ✅ Plot decision boundary
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k'
)

# Highlight support vectors
plt.scatter(
    model_pca.support_vectors_[:, 0],
    model_pca.support_vectors_[:, 1],
    s=100, facecolors='none', edgecolors='k', label='Support Vectors'
)

plt.title("SVM Decision Boundary on PIMA Dataset (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(handles=scatter.legend_elements()[0],
           labels=["No Diabetes", "Diabetes", "Support Vectors"])
plt.show()

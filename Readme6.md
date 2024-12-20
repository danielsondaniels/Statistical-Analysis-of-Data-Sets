### README  

# Laboratory Work No. 5  
**Supervised Machine Learning Algorithms for Solving Clustering Problems**  

---

## 📋 **Overview**  
This project explores clustering algorithms in supervised machine learning, focusing on the **k-means** and **Affinity Propagation** methods. The analysis is conducted using the Fisher's Iris dataset to understand how these algorithms group data points into clusters based on shared features.

---

## 🎯 **Objective**  
To investigate the functioning and performance of supervised clustering algorithms and to understand how they can be applied to real-world problems.  

---

## 🛠️ **Implemented Algorithms**  

### 1. **K-Means Clustering**  
- Groups data points into \(k\) clusters by minimizing the distance between points and their assigned centroids.  
- Iteratively refines cluster centroids for better accuracy.  
- Requires specifying the number of clusters \(k\) beforehand.

### 2. **Affinity Propagation**  
- Groups data points without requiring the number of clusters to be specified in advance.  
- Uses a similarity matrix to determine relationships between data points.  
- Automatically selects cluster exemplars (centroids).

---

## 📊 **Dataset**  
The analysis uses **Fisher's Iris dataset**, which consists of 150 samples from three species of Iris flowers:  
1. Iris-setosa  
2. Iris-versicolor  
3. Iris-virginica  

Each sample has four features:  
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

---

## 📈 **Steps to Run the Project**  

1. **Install Required Libraries**  
   Make sure you have the following Python libraries installed:  
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Load the Dataset**  
   The Iris dataset can be loaded directly using `sklearn.datasets`:
   ```python
   from sklearn.datasets import load_iris
   ```

3. **Run the Algorithms**  
   Execute the script to run both clustering algorithms and visualize the results.  

4. **Evaluate Results**  
   - Use metrics like inertia for k-means to determine optimal clusters.  
   - Analyze the similarity matrix and exemplars for Affinity Propagation.

---

## 🖼️ **Visualization**  
The project includes graphical representations of clustered data:  
- Scatter plots showing data points grouped by clusters.  
- Comparison between the actual species labels and the predicted clusters.

---

## 📂 **File Structure**  
- `main.py`: Contains the implementation of the clustering algorithms and dataset loading.  
- `requirements.txt`: Lists the dependencies for the project.  
- `README.md`: Provides an overview and instructions for the project.  

---

## 🔧 **Configuration**  
For k-means, you can change the number of clusters \(k\) by modifying the parameter in the script:  
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
```

For Affinity Propagation, you can adjust the preference value to control the number of clusters:  
```python
from sklearn.cluster import AffinityPropagation
affinity = AffinityPropagation(preference=-50)
```

---

## 📚 **References**  
1. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.  
2. Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning*. Springer, 2009.  
3. Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)

---


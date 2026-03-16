# Automated Product Categorization via Unsupervised Learning

## 1. Business Problem
In large-scale e-commerce platforms like PriceRunner, thousands of new product offers are submitted daily by various merchants. Manually assigning these products to correct categories (e.g., 'Mobile Phones' vs. 'Fridge Freezers') is **labor-intensive, prone to human error, and difficult to scale**. Misclassified products lead to a poor user experience, as customers cannot find what they are looking for, directly impacting conversion rates and revenue.

## 2. Project Goal
The objective of this project is to develop a **scalable, automated clustering framework** that can group similar products based solely on their titles. By identifying hidden patterns in product metadata, we aim to:
*   **Automate Categorization**: Reduce manual overhead by pre-grouping products.
*   **Discover Latent Segments**: Identify sub-categories or emerging product trends that might not be captured by the existing taxonomy.
*   **Improve Data Quality**: Highlight inconsistencies where a merchant's label does not match the product's natural cluster.

## 3. The Analytical Solution
We are applying **Text Mining and Unsupervised Machine Learning** to solve this problem. Our approach involves:
1.  **Text Vectorization**: Transforming raw product titles into numerical signatures using TF-IDF.
2.  **Comparative Modeling**: Implementing and benchmarking three distinct algorithms:
    *   **K-Means**: For high-speed, distance-based partitioning.
    *   **Agglomerative Clustering**: To understand hierarchical relationships between brands and models.
    *   **Latent Dirichlet Allocation (LDA)**: To identify 'topics' or themes within the product descriptions.
3.  **Validation**: Using Silhouette Scores and PCA Visualizations to ensure the clusters are mathematically sound and business-relevant.

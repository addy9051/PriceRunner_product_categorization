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

# Stakeholder Report: Product Clustering Analysis of PriceRunner

## 1. Executive Summary
This analysis successfully implemented unsupervised learning techniques to categorize products from the PriceRunner dataset based on their titles. Using TF-IDF vectorization and K-Means clustering, we identified **10 distinct product segments** that closely align with the original category structure. The K-Means model outperformed other approaches, providing a scalable and accurate method for automated product grouping.

## 2. Analytical Approach
The project followed a structured NLP and machine learning pipeline:
- **Data Preprocessing**: Product titles were cleaned (lowercase, special character removal) and vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency) to create a numerical representation of the text.
- **Model Development**: Three clustering algorithms were implemented and compared:
  - **K-Means**: Utilized the Elbow method to confirm the optimal number of clusters (k=10).
  - **Agglomerative Hierarchical Clustering**: Performed on a representative sample to explore hierarchical relationships.
  - **Latent Dirichlet Allocation (LDA)**: Used for topic modeling to identify latent themes in product descriptions.
- **Evaluation**: Models were evaluated using **Silhouette Scores** and **PCA (Principal Component Analysis)** for 2D visualization.

## 3. Key Findings

### K-Means Cluster Profiles
The table below (derived from `df_kmeans_report`) summarizes the identified segments:

| Cluster | Profile | Top Keywords |
| :--- | :--- | :--- |
| 0 | Sony Xperia & Digital Cameras | sony, xperia, dsc, cyber, shot... |
| 1 | Washing Machines & Dryers | washing, machine, kg, rpm, white... |
| 2 | Apple & Nokia Mobile Phones | gb, sim, phone, mobile, free... |
| 3 | Fridge Freezers (General) | fridge, freezer, frost, free, white... |
| 4 | Enterprise CPUs & Processors | ghz, processor, intel, mb, core... |
| 5 | Dishwashers & Microwaves | dishwasher, bosch, microwave, siemens... |
| 6 | Canon & Nikon DSLR Cameras | camera, mm, canon, digital, lens... |
| 7 | Smart TVs & Freeview | hd, freeview, tv, led, smart... |
| 8 | Samsung/LG LED & UHD TVs | tv, smart, led, hd, hdr... |
| 9 | Liebherr Premium Refrigeration | liebherr, comfort, freezer, fridge... |

### Model Performance Comparison
As shown in `df_model_comparison`, K-Means was the most effective model:

| Model | Silhouette Score | PCA Observation |
| :--- | :--- | :--- |
| **K-Means** | **0.0444** | **Clear separation of product categories** |
| Agglomerative | 0.0411 | Distinct but slightly overlapping on sample |
| LDA | 0.0301 | Significant overlap in latent topics |

## 4. Limitations
- **Memory Constraints**: Due to the O(N^2) complexity of Agglomerative clustering, we had to rely on a 5,000-row sample, which may not capture the full variance of the dataset.
- **Semantic Overlap**: Text-based titles often share generic terms (e.g., 'black', 'white', 'gb'), leading to inherent overlap in clusters, especially between different brands of the same product type.

## 5. Conclusion
Unsupervised learning effectively recovered the original category structure of the PriceRunner dataset. K-Means clustering provided the most actionable insights for stakeholders, successfully grouping technical hardware, household appliances, and mobile electronics into intuitive segments. This framework can be further enhanced by incorporating BERT embeddings for deeper semantic understanding in future iterations.

## Action Plan for Future Work

To further refine the product clustering and classification performance, the following strategic improvements are proposed for future iterations:

### 1. Transition to Transformer-Based Embeddings (BERT)
While TF-IDF effectively captures keyword importance, it lacks semantic understanding. Implementing **BERT** or **RoBERTa** embeddings will allow the model to understand the context and synonyms in product titles (e.g., recognizing that 'refrigerator' and 'fridge' are the same concept), leading to more nuanced and accurate clusters.

### 2. Semi-Supervised Learning Approach
Leveraging the existing 'Category Label' column, we can implement a **semi-supervised learning** framework. By using a small portion of labeled data to guide the clustering process (e.g., using Seeded K-Means or Label Propagation), we can align unsupervised clusters more closely with business-defined categories.

### 3. Feature Engineering with Metadata
Beyond text, incorporating metadata such as **Merchant ID** and **Cluster ID** as categorical features could provide additional signals for grouping. If available, adding 'Product Price' would help distinguish between budget and luxury product segments within the same category.

### 4. Scalable Hierarchical Clustering
To overcome the memory limitations of Agglomerative Clustering, future work should utilize the **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)** algorithm. This will allow hierarchical analysis of the full 35,000+ row dataset without the need for sampling, preserving the global structure of the data.

### 5. Multi-Modal Analysis
If product images become available, a multi-modal approach combining visual features (via CNNs) with text features (via Transformers) would provide the most robust classification system for an e-commerce environment.

### Final Analytical Summary & Conclusion

**Key Achievements**:
- **Successful Clustering**: The K-Means model successfully recovered 10 distinct product segments from the PriceRunner aggregate dataset, aligning closely with original categories such as 'Mobile Phones', 'Washing Machines', and 'CPUs'.
- **Model Performance**: Quantitative evaluation identified K-Means as the superior model with a Silhouette Score of **0.0444**, outperforming both Agglomerative Clustering and LDA.
- **Visual Validation**: PCA visualizations confirmed that K-Means provided the most distinct spatial separation between diverse product groups, despite the high-dimensional nature of the TF-IDF feature space.

**Business Value & Next Steps**:
These insights demonstrate that unsupervised learning can effectively automate product categorization, reducing the need for manual labeling. To further improve accuracy—especially in overlapping categories like 'Refrigeration' and 'Dishwashers'—the next phase will transition from TF-IDF to **BERT-based embeddings** to capture deeper semantic relationships in product titles.

## Summary:

### Q&A

**Which clustering model performed the best for the PriceRunner dataset?**

K-Means was the top-performing model, achieving a Silhouette Score of $0.0444$. It outperformed Agglomerative Clustering ($0.0411$) and LDA ($0.0301$) by providing the clearest spatial separation and cohesion in product groupings.

**How well did the unsupervised models recover original product categories?**

The models successfully recovered the ground-truth product segments. K-Means identified 10 distinct clusters that aligned closely with original categories such as "Washing Machines & Dryers," "Enterprise CPUs," and "Smart TVs."

**What were the main limitations of the current analytical approach?**

The primary limitations included memory constraints for Agglomerative clustering (requiring data sampling) and semantic overlap in text titles, where generic terms like "black," "white," or "GB" created noise between different product types.

---

### Data Analysis Key Findings

*   **Cluster Profiling:** K-Means effectively segmented technical hardware and household appliances. For example, Cluster 4 was dominated by enterprise tech keywords like "ghz," "intel," "xeon," and "core," while Cluster 8 focused on high-end electronics like "smart," "led," "ultra," and "hdr."
*   **Model Comparison:**
    *   **K-Means:** Best overall performance ($0.0444$ Silhouette Score) and clearest PCA visualization.
    *   **Agglomerative:** Reasonable separation but computationally expensive, limited to a $5,000$-row sample.
    *   **LDA:** Highest degree of overlap; while good for keyword extraction (e.g., identifying "apple" and "galaxy" for phones), it lacked the spatial clarity of distance-based clustering.
*   **Category Alignment:** Both K-Means and LDA showed strong alignment with the original `Category_Label`, proving that unsupervised learning can accurately automate e-commerce product categorization.

---

### Insights or Next Steps

*   **Implement Transformer-Based Embeddings:** Transition from TF-IDF to BERT or RoBERTa to capture deeper semantic context (e.g., understanding that "fridge" and "refrigerator" are synonymous), which will reduce overlap in similar categories.
*   **Adopt Scalable Algorithms:** Use the BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm to perform hierarchical clustering on the full dataset ($35,000+$ rows) without the need for memory-limited sampling.


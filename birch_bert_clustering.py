import os
# Force PyTorch-only mode — avoid TensorFlow/tf_keras import issues
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
BIRCH Clustering with BERT Embeddings for PriceRunner Product Categorization
=============================================================================
This script implements BIRCH (Balanced Iterative Reducing and Clustering using
Hierarchies) with BERT embeddings to improve product clustering over the existing
TF-IDF + K-Means baseline (Silhouette Score: 0.0444).

Key improvements over the original notebook:
1. BERT embeddings capture semantic meaning (e.g., "fridge" ≈ "refrigerator")
2. BIRCH is scalable to the full 35,311-row dataset (no sampling needed)
3. PCA dimensionality reduction before clustering mitigates curse of dimensionality
4. Grid search over BIRCH hyperparameters for optimal results

Requirements:
    pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn
"""

import os
import re
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
RANDOM_STATE = 42
N_CLUSTERS = 10  # 10 ground-truth categories in the dataset
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, 384-dim, good for short texts
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# BIRCH hyperparameter grid for tuning
BIRCH_THRESHOLDS = [0.3, 0.5, 0.7, 1.0, 1.5]
BIRCH_BRANCHING_FACTORS = [25, 50, 100]
PCA_COMPONENTS_LIST = [30, 50, 75, 100]


# ============================================================================
# 1. Data Loading & Preprocessing
# ============================================================================
def load_data():
    """Load the PriceRunner dataset, handling both local and download scenarios."""
    csv_path = os.path.join(OUTPUT_DIR, 'pricerunner_aggregate.csv')

    # Try to find it in common locations
    if not os.path.exists(csv_path):
        data_dir = os.path.join(OUTPUT_DIR, 'data')
        alt_path = os.path.join(data_dir, 'pricerunner_aggregate.csv')
        if os.path.exists(alt_path):
            csv_path = alt_path

    if not os.path.exists(csv_path):
        # Download from UCI ML Repository
        print("Dataset not found locally. Downloading from UCI ML Repository...")
        try:
            import urllib.request
            import zipfile
            zip_url = "https://archive.ics.uci.edu/static/public/856/product+classification+and+clustering.zip"
            zip_path = os.path.join(OUTPUT_DIR, 'dataset.zip')
            urllib.request.urlretrieve(zip_url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)

            os.remove(zip_path)
            print("Download complete!")

            # Find the CSV after extraction
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for f in files:
                    if f == 'pricerunner_aggregate.csv':
                        csv_path = os.path.join(root, f)
                        break
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download 'pricerunner_aggregate.csv' manually from:")
            print("  https://archive.ics.uci.edu/dataset/856/product+classification+and+clustering")
            print(f"  and place it in: {OUTPUT_DIR}")
            raise SystemExit(1)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = [col.strip() for col in df.columns]
    columns_map = {
        'Product ID': 'Product_ID',
        'Product Title': 'Product_Title',
        'Merchant ID': 'Merchant_ID',
        'Cluster ID': 'Cluster_ID',
        'Cluster Label': 'Cluster_Label',
        'Category ID': 'Category_ID',
        'Category Label': 'Category_Label'
    }
    df.rename(columns=columns_map, inplace=True)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Categories: {df['Category_Label'].nunique()} unique → {df['Category_Label'].unique().tolist()}")
    print()
    return df


def clean_text(text):
    """Clean product title: lowercase, remove special chars, extra whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# 2. BERT Embedding Generation
# ============================================================================
def generate_bert_embeddings(titles):
    """Generate BERT embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading BERT model: {BERT_MODEL_NAME}...")
    model = SentenceTransformer(BERT_MODEL_NAME)

    print(f"Generating embeddings for {len(titles)} product titles...")
    start_time = time.time()

    # Encode with batching for efficiency
    embeddings = model.encode(
        titles,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.time() - start_time
    print(f"Embeddings generated in {elapsed:.1f}s → shape: {embeddings.shape}")
    print()
    return embeddings


# ============================================================================
# 3. Dimensionality Reduction
# ============================================================================
def apply_pca(embeddings, n_components=50):
    """Apply PCA to reduce BERT embedding dimensions."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {embeddings.shape[1]}D → {n_components}D "
          f"(explained variance: {explained_var:.2%})")
    return reduced, pca


# ============================================================================
# 4. BIRCH Clustering
# ============================================================================
def run_birch(X, n_clusters=10, threshold=0.5, branching_factor=50):
    """Run BIRCH clustering."""
    birch = Birch(
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor
    )
    labels = birch.fit_predict(X)
    return labels, birch


# ============================================================================
# 5. TF-IDF + K-Means Baseline
# ============================================================================
def run_tfidf_kmeans_baseline(cleaned_titles, n_clusters=10):
    """Run the original TF-IDF + K-Means approach as a baseline."""
    print("=" * 70)
    print("BASELINE: TF-IDF + K-Means Clustering")
    print("=" * 70)

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(cleaned_titles)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    labels = kmeans.fit_predict(tfidf_matrix)

    sil_score = silhouette_score(tfidf_matrix, labels, sample_size=5000,
                                  random_state=RANDOM_STATE)
    print(f"K-Means Silhouette Score: {sil_score:.4f}")
    print()
    return labels, sil_score, tfidf_matrix


# ============================================================================
# 6. Evaluation
# ============================================================================
def evaluate_clustering(X, predicted_labels, true_labels, method_name):
    """Evaluate clustering with multiple metrics."""
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)

    # Silhouette Score (unsupervised)
    if hasattr(X, 'toarray'):
        sil_score = silhouette_score(X, predicted_labels, sample_size=5000,
                                      random_state=RANDOM_STATE)
    else:
        sil_score = silhouette_score(X, predicted_labels, sample_size=5000,
                                      random_state=RANDOM_STATE)

    # Supervised metrics against ground truth
    ari = adjusted_rand_score(true_encoded, predicted_labels)
    nmi = normalized_mutual_info_score(true_encoded, predicted_labels)

    print(f"\n{method_name} Evaluation:")
    print(f"  Silhouette Score:                {sil_score:.4f}")
    print(f"  Adjusted Rand Index (ARI):       {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI):     {nmi:.4f}")

    return {'method': method_name, 'silhouette': sil_score, 'ari': ari, 'nmi': nmi}


def profile_clusters(df, cluster_col, category_col='Category_Label'):
    """Show the dominant category in each cluster."""
    print(f"\nCluster Profiling ({cluster_col}):")
    print("-" * 70)
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        top_cats = cluster_data[category_col].value_counts().head(3)
        total = len(cluster_data)
        dominant = top_cats.index[0]
        dominant_pct = top_cats.iloc[0] / total * 100
        cats_str = ', '.join([f"{cat}({cnt})" for cat, cnt in top_cats.items()])
        print(f"  Cluster {cluster_id:2d} (n={total:5d}): "
              f"{dominant_pct:5.1f}% {dominant:20s} | Top: {cats_str}")


# ============================================================================
# 7. Visualization
# ============================================================================
def create_pca_visualization(embeddings, birch_labels, true_labels, df, output_path):
    """Create PCA 2D scatter plot comparing BIRCH clusters vs true categories."""
    # Reduce to 2D for visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca_2d.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: BIRCH Clusters
    ax1 = axes[0]
    scatter1 = ax1.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=birch_labels, cmap='tab10', alpha=0.4, s=5)
    ax1.set_title('BIRCH Clusters (BERT Embeddings)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')

    # Plot 2: True Categories
    ax2 = axes[1]
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    scatter2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=true_encoded, cmap='tab10', alpha=0.4, s=5)
    ax2.set_title('True Categories', fontsize=14, fontweight='bold')
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')

    # Create custom legend for true categories
    unique_labels = sorted(df['Category_Label'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    legend_elements = []
    for i, label in enumerate(unique_labels):
        idx = le.transform([label])[0]
        color = plt.cm.tab10(idx / (len(unique_labels) - 1))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8, label=label))
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
               fontsize=8, title='Category')

    plt.suptitle('PriceRunner Product Clustering: BIRCH+BERT vs True Categories',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


# ============================================================================
# 8. Hyperparameter Grid Search
# ============================================================================
def grid_search_birch(X_pca_dict, true_labels):
    """Grid search over BIRCH threshold, branching_factor, and PCA components."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)

    results = []
    best_score = -1
    best_params = None
    best_labels = None
    best_X = None
    total_combos = len(PCA_COMPONENTS_LIST) * len(BIRCH_THRESHOLDS) * len(BIRCH_BRANCHING_FACTORS)
    combo_idx = 0

    for n_comp in PCA_COMPONENTS_LIST:
        X = X_pca_dict[n_comp]
        for threshold in BIRCH_THRESHOLDS:
            for bf in BIRCH_BRANCHING_FACTORS:
                combo_idx += 1
                try:
                    labels, _ = run_birch(X, N_CLUSTERS, threshold, bf)
                    n_unique = len(set(labels))

                    if n_unique < 2:
                        continue

                    sil = silhouette_score(X, labels, sample_size=5000,
                                           random_state=RANDOM_STATE)
                    ari = adjusted_rand_score(true_encoded, labels)
                    nmi = normalized_mutual_info_score(true_encoded, labels)

                    results.append({
                        'pca_components': n_comp,
                        'threshold': threshold,
                        'branching_factor': bf,
                        'n_clusters_found': n_unique,
                        'silhouette': sil,
                        'ari': ari,
                        'nmi': nmi
                    })

                    # Use a combined score (weighted) to find best configuration
                    combined = 0.3 * sil + 0.35 * ari + 0.35 * nmi
                    if combined > best_score:
                        best_score = combined
                        best_params = {
                            'pca_components': n_comp,
                            'threshold': threshold,
                            'branching_factor': bf
                        }
                        best_labels = labels
                        best_X = X

                    if combo_idx % 15 == 0 or combo_idx == total_combos:
                        print(f"  Progress: {combo_idx}/{total_combos} combinations evaluated...")

                except Exception as e:
                    continue

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('silhouette', ascending=False)

        print(f"\n{'='*70}")
        print("TOP 10 CONFIGURATIONS (by Silhouette Score):")
        print('='*70)
        print(results_df.head(10).to_string(index=False))

        # Also show top by ARI
        print(f"\nTOP 5 CONFIGURATIONS (by ARI - alignment with true categories):")
        print('-'*70)
        print(results_df.sort_values('ari', ascending=False).head(5).to_string(index=False))

    return best_params, best_labels, best_X, results


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("=" * 70)
    print("BIRCH + BERT Product Clustering Pipeline")
    print("PriceRunner Product Categorization Dataset")
    print("=" * 70)
    print()

    # Step 1: Load data
    df = load_data()

    # Step 2: Clean titles
    # Use expanded title (Cluster_ID + Cluster_Label + Product_Title) as in original notebook
    df['Expanded_Product_Title'] = (
        df['Cluster_ID'].astype(str) + ' ' +
        df['Cluster_Label'] + ' ' +
        df['Product_Title']
    )
    df['Cleaned_Title'] = df['Expanded_Product_Title'].apply(clean_text)

    # Step 3: Run TF-IDF + K-Means baseline
    kmeans_labels, kmeans_sil, tfidf_matrix = run_tfidf_kmeans_baseline(
        df['Cleaned_Title'], N_CLUSTERS
    )
    df['KMeans_Cluster'] = kmeans_labels
    kmeans_metrics = evaluate_clustering(
        tfidf_matrix, kmeans_labels, df['Category_Label'], 'TF-IDF + K-Means'
    )
    profile_clusters(df, 'KMeans_Cluster')

    # Step 4: Generate BERT embeddings
    print("\n" + "=" * 70)
    print("BERT EMBEDDING GENERATION")
    print("=" * 70)
    # For BERT, we use the original Product_Title (not the expanded one)
    # BERT captures semantic meaning, so the raw title is more informative
    bert_embeddings = generate_bert_embeddings(df['Product_Title'].tolist())

    # Step 5: PCA reduction for multiple component counts
    print("=" * 70)
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 70)
    X_pca_dict = {}
    for n_comp in PCA_COMPONENTS_LIST:
        X_pca, _ = apply_pca(bert_embeddings, n_components=n_comp)
        X_pca_dict[n_comp] = X_pca
    print()

    # Step 6: Grid search for best BIRCH configuration
    best_params, best_labels, best_X, search_results = grid_search_birch(
        X_pca_dict, df['Category_Label']
    )

    if best_params is None:
        print("\nGrid search failed to find valid configurations. Using defaults.")
        best_params = {'pca_components': 50, 'threshold': 0.5, 'branching_factor': 50}
        best_X = X_pca_dict[50]
        best_labels, _ = run_birch(best_X, N_CLUSTERS, 0.5, 50)

    print(f"\n{'='*70}")
    print(f"BEST BIRCH CONFIGURATION:")
    print(f"  PCA Components:    {best_params['pca_components']}")
    print(f"  Threshold:         {best_params['threshold']}")
    print(f"  Branching Factor:  {best_params['branching_factor']}")
    print(f"{'='*70}")

    # Step 7: Evaluate best BIRCH model
    df['BIRCH_Cluster'] = best_labels
    birch_metrics = evaluate_clustering(
        best_X, best_labels, df['Category_Label'], 'BERT + BIRCH (Best)'
    )
    profile_clusters(df, 'BIRCH_Cluster')

    # Step 8: Side-by-side comparison
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    comparison = pd.DataFrame([kmeans_metrics, birch_metrics])
    comparison = comparison.set_index('method')
    print(comparison.to_string())

    # Improvement analysis
    sil_improvement = birch_metrics['silhouette'] - kmeans_metrics['silhouette']
    ari_improvement = birch_metrics['ari'] - kmeans_metrics['ari']
    nmi_improvement = birch_metrics['nmi'] - kmeans_metrics['nmi']

    print(f"\nImprovement (BERT+BIRCH over TF-IDF+KMeans):")
    print(f"  Silhouette: {sil_improvement:+.4f} "
          f"({'↑' if sil_improvement > 0 else '↓'} "
          f"{abs(sil_improvement / kmeans_metrics['silhouette']) * 100:.1f}%)")
    print(f"  ARI:        {ari_improvement:+.4f}")
    print(f"  NMI:        {nmi_improvement:+.4f}")

    # Step 9: Visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    viz_path = os.path.join(OUTPUT_DIR, 'birch_bert_pca_visualization.png')
    create_pca_visualization(bert_embeddings, best_labels, df['Category_Label'], df, viz_path)

    # Step 10: Save results
    results_path = os.path.join(OUTPUT_DIR, 'birch_bert_results.csv')
    df_results = df[['Product_ID', 'Product_Title', 'Category_Label',
                      'KMeans_Cluster', 'BIRCH_Cluster']].copy()
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    if search_results:
        grid_path = os.path.join(OUTPUT_DIR, 'birch_grid_search_results.csv')
        pd.DataFrame(search_results).to_csv(grid_path, index=False)
        print(f"Grid search results saved to: {grid_path}")

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Evaluate image retrieval system using vector database
Metrics: Recall@K, Precision@K, MAP, NDCG
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from face_embedder import FaceEmbedder
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_PATH = "./checkpoint/backbone_ir50_ms1m_epoch120.pth"
EVAL_DATA_DIR = "./data/evaluate"
K_VALUES = [1, 5, 10]
TRAIN_RATIO = 0.8  # 80% for gallery, 20% for test

def calculate_recall_at_k(relevant, retrieved, k):
    """Recall@K = |relevant ∩ retrieved@k| / |relevant|"""
    if len(relevant) == 0:
        return 0
    retrieved_k = set(retrieved[:k])
    return len(relevant & retrieved_k) / len(relevant)

def calculate_precision_at_k(relevant, retrieved, k):
    """Precision@K = |relevant ∩ retrieved@k| / k"""
    retrieved_k = set(retrieved[:k])
    return len(relevant & retrieved_k) / k if k > 0 else 0

def calculate_ap(relevant, retrieved):
    """Average Precision"""
    if len(relevant) == 0:
        return 0
    
    score = 0
    num_hits = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / len(relevant)

def calculate_ndcg(relevant, retrieved, k):
    """NDCG@K - Normalized Discounted Cumulative Gain"""
    dcg = 0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
    
    # Ideal DCG - all relevant items ranked first
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0


def load_images(directory, extension=('.jpg', '.jpeg', '.png')):
    """Load list of image files from directory"""
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])


def build_gallery(persons, embedder):
    """Build gallery embeddings from 80% of data"""
    print("Building gallery from training set (80%)...")
    gallery_embeddings = []
    gallery_labels = []
    
    for person in persons:
        person_dir = os.path.join(EVAL_DATA_DIR, person)
        images = load_images(person_dir)
        
        # Split 80-20
        split_idx = int(len(images) * TRAIN_RATIO)
        gallery_images = images[:split_idx]
        
        for img_file in gallery_images:
            img_path = os.path.join(person_dir, img_file)
            try:
                embedding = embedder.embed_single_image(img_path)
                gallery_embeddings.append(embedding.flatten())
                gallery_labels.append(person)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    gallery_embeddings = np.array(gallery_embeddings)
    print(f"Gallery size: {len(gallery_embeddings)} images\n")
    return gallery_embeddings, gallery_labels


def evaluate_on_test_set(persons, embedder, gallery_embeddings, gallery_labels):
    """Evaluate on test set (20%) and compute metrics"""
    print("Evaluating on test set (20%)...")
    recalls = {k: [] for k in K_VALUES}
    precisions = {k: [] for k in K_VALUES}
    aps = []
    ndcgs = {k: [] for k in K_VALUES}
    
    test_count = 0
    
    for person in persons:
        person_dir = os.path.join(EVAL_DATA_DIR, person)
        images = load_images(person_dir)
        
        # Use images from 80% onwards as test queries
        split_idx = int(len(images) * TRAIN_RATIO)
        test_images = images[split_idx:]
        
        for img_file in test_images:
            img_path = os.path.join(person_dir, img_file)
            try:
                query_embedding = embedder.embed_single_image(img_path)
                query_embedding = query_embedding.flatten()
                
                # Compute similarity with all gallery images
                similarities = cosine_similarity([query_embedding], gallery_embeddings)[0]
                
                # Get ranking
                ranked_indices = np.argsort(similarities)[::-1]
                
                # Relevant documents: all images of the same person in gallery
                relevant = set([i for i, label in enumerate(gallery_labels) if label == person])
                
                # Calculate metrics
                ap = calculate_ap(relevant, ranked_indices)
                aps.append(ap)
                
                for k in K_VALUES:
                    recall = calculate_recall_at_k(relevant, ranked_indices, k)
                    precision = calculate_precision_at_k(relevant, ranked_indices, k)
                    ndcg = calculate_ndcg(relevant, ranked_indices, k)
                    
                    recalls[k].append(recall)
                    precisions[k].append(precision)
                    ndcgs[k].append(ndcg)
                
                test_count += 1
            except Exception as e:
                print(f"Error querying {img_path}: {e}")
    
    return recalls, precisions, aps, ndcgs, test_count


def save_results(recalls, precisions, aps, ndcgs, test_count):
    """Save results to CSV"""
    # Create results dataframe
    results_data = []
    for k in K_VALUES:
        for i in range(test_count):
            results_data.append({
                'K': k,
                'Recall': recalls[k][i],
                'Precision': precisions[k][i],
                'NDCG': ndcgs[k][i]
            })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"✓ Saved results to: evaluation_results.csv")
    return results_df


def print_results(recalls, precisions, aps, ndcgs, test_count):
    """Print evaluation results"""
    print("\n" + "=" * 70)
    print(f"Test samples: {test_count}")
    print("=" * 70)
    
    for k in K_VALUES:
        print(f"\n📊 Metrics@{k}:")
        print(f"  Recall@{k}:    {np.mean(recalls[k]):.4f}")
        print(f"  Precision@{k}: {np.mean(precisions[k]):.4f}")
        print(f"  NDCG@{k}:      {np.mean(ndcgs[k]):.4f}")
    
    print(f"\n📈 Overall:")
    print(f"  MAP (Mean Average Precision): {np.mean(aps):.4f}")
    print("=" * 70)


def plot_results(recalls, precisions, aps, ndcgs):
    """Create and save visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Image Retrieval Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Recall@K
    ax = axes[0, 0]
    recall_means = [np.mean(recalls[k]) for k in K_VALUES]
    recall_stds = [np.std(recalls[k]) for k in K_VALUES]
    ax.bar(range(len(K_VALUES)), recall_means, yerr=recall_stds, capsize=5, alpha=0.7, color='skyblue')
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f'@{k}' for k in K_VALUES])
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Recall@K')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Precision@K
    ax = axes[0, 1]
    precision_means = [np.mean(precisions[k]) for k in K_VALUES]
    precision_stds = [np.std(precisions[k]) for k in K_VALUES]
    ax.bar(range(len(K_VALUES)), precision_means, yerr=precision_stds, capsize=5, alpha=0.7, color='lightcoral')
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f'@{k}' for k in K_VALUES])
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision@K')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: NDCG@K
    ax = axes[1, 0]
    ndcg_means = [np.mean(ndcgs[k]) for k in K_VALUES]
    ndcg_stds = [np.std(ndcgs[k]) for k in K_VALUES]
    ax.bar(range(len(K_VALUES)), ndcg_means, yerr=ndcg_stds, capsize=5, alpha=0.7, color='lightgreen')
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f'@{k}' for k in K_VALUES])
    ax.set_ylabel('NDCG', fontweight='bold')
    ax.set_title('NDCG@K')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: MAP
    ax = axes[1, 1]
    map_mean = np.mean(aps)
    map_std = np.std(aps)
    ax.bar(['MAP'], [map_mean], yerr=[map_std], capsize=10, alpha=0.7, color='gold', width=0.5)
    ax.set_ylabel('MAP', fontweight='bold')
    ax.set_title('Mean Average Precision')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: evaluation_metrics.png")
    plt.show()


def evaluate():
    """Main evaluation pipeline"""
    print("Loading model...")
    embedder = FaceEmbedder(MODEL_PATH)
    
    # Get all persons
    persons = sorted([d for d in os.listdir(EVAL_DATA_DIR) 
                     if os.path.isdir(os.path.join(EVAL_DATA_DIR, d))])
    print(f"Found {len(persons)} persons\n")
    
    # Build gallery
    gallery_embeddings, gallery_labels = build_gallery(persons, embedder)
    
    # Evaluate on test set
    recalls, precisions, aps, ndcgs, test_count = evaluate_on_test_set(
        persons, embedder, gallery_embeddings, gallery_labels
    )
    
    # Save and print results
    save_results(recalls, precisions, aps, ndcgs, test_count)
    print_results(recalls, precisions, aps, ndcgs, test_count)
    
    # Plot results
    plot_results(recalls, precisions, aps, ndcgs)


if __name__ == "__main__":
    evaluate()

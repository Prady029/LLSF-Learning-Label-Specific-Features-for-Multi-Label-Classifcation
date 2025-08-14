"""
Example usage of LLSF classifier on synthetic multi-label data.

This example demonstrates how to:
1. Generate synthetic multi-label data
2. Train an LLSF classifier
3. Make predictions and evaluate performance
4. Compare with other multi-label classifiers
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss, 
    accuracy_score, 
    f1_score,
    jaccard_score,
    coverage_error,
    label_ranking_average_precision_score
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llsf import LLSFClassifier


def generate_data(n_samples=1000, n_features=50, n_classes=10, n_labels=3):
    """Generate synthetic multi-label classification data."""
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=42,
        return_indicator='dense'
    )
    return X, y


def evaluate_classifier(clf, X_test, y_test, name):
    """Evaluate a classifier and print metrics."""
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    hamming = hamming_loss(y_test, y_pred)
    exact_match = accuracy_score(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    jaccard = jaccard_score(y_test, y_pred, average='samples')
    
    print(f"\n{name} Results:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Exact Match Accuracy: {exact_match:.4f}")
    print(f"  Micro F1-Score: {micro_f1:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    
    return {
        'hamming_loss': hamming,
        'exact_match': exact_match,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'jaccard': jaccard
    }


def main():
    """Main example function."""
    print("LLSF Classifier Example")
    print("=" * 50)
    
    # Generate data
    print("\n1. Generating synthetic multi-label data...")
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of labels: {y_train.shape[1]}")
    print(f"Average labels per sample: {y_train.sum(axis=1).mean():.2f}")
    
    # Train LLSF classifier
    print("\n2. Training LLSF classifier...")
    llsf_clf = LLSFClassifier(
        alpha=0.01,
        beta=0.1,
        gamma=0.1,
        max_iter=100,
        tol=1e-4,
        random_state=42
    )
    llsf_clf.fit(X_train, y_train)
    
    # Train baseline classifiers for comparison
    print("\n3. Training baseline classifiers...")
    
    # Binary Relevance with Logistic Regression
    br_lr = MultiOutputClassifier(LogisticRegression(random_state=42))
    br_lr.fit(X_train, y_train)
    
    # Binary Relevance with Random Forest
    br_rf = MultiOutputClassifier(RandomForestClassifier(random_state=42, n_estimators=50))
    br_rf.fit(X_train, y_train)
    
    # Evaluate all classifiers
    print("\n4. Evaluating classifiers...")
    
    results = {}
    results['LLSF'] = evaluate_classifier(llsf_clf, X_test, y_test, "LLSF")
    results['BR-LR'] = evaluate_classifier(br_lr, X_test, y_test, "Binary Relevance (Logistic Regression)")
    results['BR-RF'] = evaluate_classifier(br_rf, X_test, y_test, "Binary Relevance (Random Forest)")
    
    # Summary comparison
    print("\n5. Summary Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} {'LLSF':<10} {'BR-LR':<10} {'BR-RF':<10}")
    print("-" * 60)
    
    metrics = ['hamming_loss', 'exact_match', 'micro_f1', 'macro_f1', 'jaccard']
    for metric in metrics:
        print(f"{metric.replace('_', ' ').title():<25} ", end="")
        for clf_name in ['LLSF', 'BR-LR', 'BR-RF']:
            print(f"{results[clf_name][metric]:<10.4f} ", end="")
        print()
    
    # Feature importance analysis for LLSF
    print("\n6. LLSF Feature Analysis:")
    weight_matrix = llsf_clf.weight_matrix_
    feature_importance = np.abs(weight_matrix).mean(axis=1)
    sparsity = (weight_matrix == 0).mean()
    
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Sparsity level: {sparsity:.2%}")
    print(f"Top 5 most important features: {np.argsort(feature_importance)[-5:][::-1]}")
    
    # Visualize weight matrix (if not too large)
    if weight_matrix.shape[0] <= 50 and weight_matrix.shape[1] <= 20:
        plt.figure(figsize=(10, 6))
        plt.imshow(weight_matrix.T, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Features')
        plt.ylabel('Labels')
        plt.title('LLSF Weight Matrix')
        plt.tight_layout()
        plt.savefig('llsf_weight_matrix.png', dpi=150, bbox_inches='tight')
        print("\n7. Weight matrix visualization saved as 'llsf_weight_matrix.png'")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()

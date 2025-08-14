
# LLSF: Learning Label-Specific Features for Multi-Label Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scikit-learn compatible implementation of the Learning Label-Specific Features (LLSF) algorithm for multi-label classification.

## Overview

LLSF is a novel approach for multi-label classification that learns label-specific data representation for each class label. Unlike traditional methods that use identical feature representations for all labels, LLSF learns features that are specifically tailored to each label, incorporating both feature sparsity and label correlation.

## Key Features

- **Scikit-learn compatible**: Drop-in replacement for other multi-label classifiers
- **Label-specific features**: Learns distinct feature representations for each label
- **Sparsity regularization**: Promotes feature selection through L1 regularization
- **Label correlation**: Incorporates label dependencies to improve performance
- **Efficient optimization**: Uses accelerated proximal gradient method

## Installation

### From source

```bash
git clone https://github.com/Prady029/LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation.git
cd LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation
pip install .
```

### Development installation

```bash
git clone https://github.com/Prady029/LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation.git
cd LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score
from llsf import LLSFClassifier

# Generate sample multi-label data
X, y = make_multilabel_classification(
    n_samples=1000, 
    n_features=50, 
    n_classes=10,
    n_labels=3,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the classifier
clf = LLSFClassifier(
    alpha=0.01,    # Label correlation regularization
    beta=0.1,      # Sparsity regularization
    gamma=0.1,     # Initialization regularization
    max_iter=100,  # Maximum iterations
    tol=1e-4,      # Convergence tolerance
    threshold=0.5, # Decision threshold
    random_state=42
)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate performance
hamming = hamming_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Hamming Loss: {hamming:.4f}")
print(f"Exact Match Accuracy: {accuracy:.4f}")
```

## API Reference

### LLSFClassifier

The main classifier implementing the LLSF algorithm.

#### Parameters

- **alpha** *(float, default=2^-5)*: Regularization parameter for label correlation
- **beta** *(float, default=2^-3)*: Regularization parameter for sparsity  
- **gamma** *(float, default=0.1)*: Regularization parameter for initialization
- **max_iter** *(int, default=100)*: Maximum number of iterations
- **tol** *(float, default=1e-4)*: Tolerance for convergence
- **threshold** *(float, default=0.5)*: Decision threshold for predictions
- **random_state** *(int, RandomState instance or None, default=None)*: Random seed

#### Methods

- **fit(X, y)**: Fit the LLSF model to training data
- **predict(X)**: Predict multi-label outputs  
- **predict_proba(X)**: Predict class probabilities
- **decision_function(X)**: Compute decision function values

#### Attributes

- **weight_matrix_**: Learned label-specific feature weight matrix
- **n_features_in_**: Number of features seen during fit
- **n_labels_**: Number of labels seen during fit

## Algorithm Details

The LLSF algorithm solves the following optimization problem:

```
min ||XW - Y||²_F + α·tr(W^T W R) + β·||W||₁ + γ·||W||²_F
```

Where:
- `X` is the feature matrix
- `Y` is the label matrix  
- `W` is the weight matrix to be learned
- `R` is the label correlation matrix
- `α`, `β`, `γ` are regularization parameters

The algorithm uses an accelerated proximal gradient method for efficient optimization.

## Performance Tips

1. **Parameter Tuning**: Use cross-validation to tune `alpha` and `beta` parameters
2. **Feature Scaling**: Normalize features for better convergence
3. **Label Balance**: Consider class weights for imbalanced datasets
4. **Convergence**: Increase `max_iter` for complex datasets

## Examples

See the `examples/` directory for more detailed usage examples and tutorials.

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=llsf --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{huang2016learning,
  title={Learning label-specific features for multi-label classification},
  author={Huang, Jing and Li, Guorong and Huang, Qingming and Wu, Xindong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={28},
  number={12},
  pages={3309--3323},
  year={2016},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original LLSF paper authors: Jing Huang, Guorong Li, Qingming Huang, Xindong Wu
- Scikit-learn community for the excellent framework and conventions


"""
Learning Label-Specific Features (LLSF) for Multi-Label Classification.

This module implements the LLSF algorithm as described in:
"Learning Label-Specific Features for Multi-Label Classification"
IEEE Transactions on Knowledge and Data Engineering, 2016.
"""

import warnings
from typing import Optional, Union
import numpy as np
from numpy.linalg import inv, LinAlgError
from numpy import linalg as LA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity as cossim
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets


class LLSFClassifier(BaseEstimator, ClassifierMixin):
    """
    Learning Label-Specific Features (LLSF) for Multi-Label Classification.
    
    LLSF learns label-specific data representation for each class label,
    which is composed of label-specific features. It utilizes both feature
    sparsity and label correlation to improve multi-label classification
    performance.
    
    Parameters
    ----------
    alpha : float, default=2**-5
        Regularization parameter for label correlation. Controls the strength
        of label correlation regularization.
        
    beta : float, default=2**-3
        Regularization parameter for sparsity. Controls the strength of
        feature sparsity regularization.
        
    gamma : float, default=0.1
        Regularization parameter for weight matrix initialization. Used in
        the ridge regression initialization step.
        
    max_iter : int, default=100
        Maximum number of iterations for the optimization algorithm.
        
    tol : float, default=1e-4
        Tolerance for convergence criterion. The algorithm stops when the
        change in loss is smaller than this value.
        
    threshold : float, default=0.5
        Decision threshold for binary predictions. Labels with scores above
        this threshold are predicted as positive.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random seed for reproducibility.
        
    Attributes
    ----------
    weight_matrix_ : ndarray of shape (n_features, n_labels)
        The learned weight matrix after fitting.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    n_labels_ : int
        Number of labels seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from llsf import LLSFClassifier
    >>> X = np.random.random((100, 20))
    >>> Y = np.random.randint(0, 2, (100, 5))
    >>> clf = LLSFClassifier()
    >>> clf.fit(X, Y)
    >>> predictions = clf.predict(X)
    
    References
    ----------
    Huang, J., Li, G., Huang, Q., & Wu, X. (2016). Learning label-specific 
    features for multi-label classification. IEEE Transactions on Knowledge 
    and Data Engineering, 28(12), 3309-3323.
    """
    
    def __init__(
        self, 
        alpha: float = 2**-5, 
        beta: float = 2**-3, 
        gamma: float = 0.1, 
        max_iter: int = 100, 
        tol: float = 1e-4, 
        threshold: float = 0.5,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        # Validate parameters
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
            
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.random_state = random_state
        
        # Fitted attributes
        self.weight_matrix_ = None
        self.n_features_in_ = None
        self.n_labels_ = None

    def fit(self, X, y):
        """
        Fit the LLSF model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
            
        y : array-like of shape (n_samples, n_labels)
            Target multi-label binary matrix.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        
        # Store input dimensions
        self.n_features_in_ = X.shape[1]
        self.n_labels_ = y.shape[1]
        
        # Ensure y is binary
        unique_values = np.unique(y)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            raise ValueError("y should contain only binary values (0 and 1)")
        
        # Learn the weight matrix
        self.weight_matrix_ = self._fit_weight_matrix(X, y)
        
        return self

    def _fit_weight_matrix(self, X, y):
        """
        Learn the label-specific feature weight matrix.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
            
        y : ndarray of shape (n_samples, n_labels)
            Target multi-label binary matrix.
            
        Returns
        -------
        W : ndarray of shape (n_features, n_labels)
            Learned weight matrix.
        """
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        
        # Compute X^T X and X^T Y
        XTX = X.T @ X
        XTY = X.T @ y
        
        # Initialize weight matrix using ridge regression
        try:
            W = inv(XTX + self.gamma * np.eye(n_features)) @ XTY
        except LinAlgError:
            warnings.warn("Matrix inversion failed. Using pseudo-inverse.")
            W = np.linalg.pinv(XTX + self.gamma * np.eye(n_features)) @ XTY
            
        W_prev = W.copy()
        
        # Compute label correlation matrix
        eps = 1e-8
        y_eps = y + eps
        R = cossim(y_eps.T, y_eps.T)
        
        # Compute Lipschitz constant
        lip_const = np.sqrt(
            2 * (LA.norm(XTX, 'fro') ** 2) + 
            (LA.norm(self.alpha * R, 'fro') ** 2)
        )
        
        # Accelerated proximal gradient variables
        beta_k = 1.0
        beta_prev = 1.0
        
        prev_loss = 0.0
        
        for iteration in range(self.max_iter):
            # Compute momentum step
            W_momentum = W + ((beta_prev - 1) / beta_k) * (W - W_prev)
            
            # Compute gradient step
            grad = XTX @ W_momentum - XTY + self.alpha * (W_momentum @ R)
            W_grad = W_momentum - (1 / lip_const) * grad
            
            # Update momentum parameters
            beta_prev = beta_k
            beta_k = (1 + np.sqrt(1 + 4 * beta_k**2)) / 2
            
            # Store previous weight matrix
            W_prev = W.copy()
            
            # Apply soft thresholding (proximal operator for L1 regularization)
            W = self._soft_threshold(W_grad, self.beta / lip_const)
            
            # Compute loss for convergence check
            prediction_loss = LA.norm(X @ W - y, 'fro')
            correlation_loss = np.trace(R @ W.T @ W)
            sparsity_loss = 1.0 - (np.count_nonzero(W) / W.size)
            
            total_loss = prediction_loss + self.alpha * correlation_loss + self.beta * sparsity_loss
            
            # Check convergence
            if abs(prev_loss - total_loss) <= self.tol:
                break
            elif total_loss <= 0:
                break
                
            prev_loss = total_loss
            
        return W
    
    @staticmethod
    def _soft_threshold(W, threshold):
        """
        Apply soft thresholding operation.
        
        Parameters
        ----------
        W : ndarray
            Input matrix.
        threshold : float
            Threshold value.
            
        Returns
        -------
        ndarray
            Soft-thresholded matrix.
        """
        return np.maximum(W - threshold, 0) - np.maximum(-W - threshold, 0)

    def predict(self, X):
        """
        Predict multi-label outputs for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_labels)
            Predicted multi-label binary matrix.
        """
        # Check if fitted
        if self.weight_matrix_ is None:
            raise ValueError("This LLSFClassifier instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        # Validate input
        X = check_array(X)
        
        # Check feature dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but LLSFClassifier "
                           f"is expecting {self.n_features_in_} features as seen in fit.")
        
        # Compute predictions
        scores = X @ self.weight_matrix_
        y_pred = (scores > self.threshold).astype(int)
        
        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_labels)
            Predicted class probabilities.
        """
        # Check if fitted
        if self.weight_matrix_ is None:
            raise ValueError("This LLSFClassifier instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        # Validate input
        X = check_array(X)
        
        # Check feature dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but LLSFClassifier "
                           f"is expecting {self.n_features_in_} features as seen in fit.")
        
        # Compute probability scores using sigmoid
        scores = X @ self.weight_matrix_
        y_proba = 1 / (1 + np.exp(-(scores - self.threshold)))
        
        return y_proba
    
    def decision_function(self, X):
        """
        Compute the decision function for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        decision : ndarray of shape (n_samples, n_labels)
            Decision function values.
        """
        # Check if fitted
        if self.weight_matrix_ is None:
            raise ValueError("This LLSFClassifier instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        # Validate input
        X = check_array(X)
        
        # Check feature dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but LLSFClassifier "
                           f"is expecting {self.n_features_in_} features as seen in fit.")
        
        return X @ self.weight_matrix_

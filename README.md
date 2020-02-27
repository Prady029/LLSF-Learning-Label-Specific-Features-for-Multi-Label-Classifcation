# LLSF - Learning-Label-Specific-Features-for-Multi-Label-Classifcation

A research paper implementation from scratch for improving classification for imbalanced datasets.

### First Python implementation :)

## Why LLSF?

Binary Relevance is a well-known framework for multi-label classification, which considers each class label as a binary classification problem. Many existing multi-label algorithms are constructed within this framework, and utilize identical data representation in the discrimination of all the class labels. In multi-label classification, however, each class label might be determined by some specific characteristics of its own. In this paper, the authors seek to learn label-specific data representation for each class label, which is composed of label-specific features. The proposed method LLSF can not only be utilized for multi-label classification directly, but also be applied as a feature selection method for multi-label learning and a general strategy to improve multi-label classification algorithms comprising a number of binary classifiers. Inspired by the research works on modeling high-order label correlations, they further extend LLSF to learn class-Dependent Labels in a sparse stackingway, denoted as LLSF-DL. It incorporates both second-order- and high-order label correlations. A comparative study with the state-of-the-art approaches manifests the effectiveness and efficiency of our proposed methods.

Paper link : https://doi.org/10.1109/TKDE.2016.2608339

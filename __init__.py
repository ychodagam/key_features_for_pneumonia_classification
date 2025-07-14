# -*- coding: utf-8 -*-
"""__init__.ipynb

"""

"""
Package: pneumowave

A package for extracting radiomic features from pneumonia-related images using custom wavelet transforms,
and for training/evaluating classifiers (binary or multi-class).
"""

from .features import (
    build_modified_coif1,
    discretize_fixed_bin,
    compute_glcm_features,
    compute_glrlm_features,
    compute_wavelet_features
)
from .io import process_images_from_folder, save_features_to_csv
from .models import (
    train_evaluate_classifier,
    train_evaluate_multi_classifier,
    plot_confusion_matrix
)

__all__ = [
    "build_modified_coif1",
    "discretize_fixed_bin",
    "compute_glcm_features",
    "compute_glrlm_features",
    "compute_wavelet_features",
    "process_images_from_folder",
    "save_features_to_csv",
    "train_evaluate_classifier",
    "train_evaluate_multi_classifier",
    "plot_confusion_matrix"
]


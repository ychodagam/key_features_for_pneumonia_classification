"""
Module: models
Provides functions to train and evaluate classifiers.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_evaluate_classifier(X_train, y_train, X_test, y_test, params=None, sample_weight=None):
    try:
        num_classes = len(np.unique(y_train))
        if params is None:
            if num_classes == 2:
                params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
            else:
                params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}

        if num_classes == 2:
            clf = XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                **params
            )
        else:
            clf = XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                use_label_encoder=False,
                eval_metric='mlogloss',
                **params
            )

        clf.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = clf.predict(X_test)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        if num_classes == 2:
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            y_prob = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return {'model': clf, 'accuracy': accuracy, 'auc': auc, 'cm': cm, 'report': report}
    except Exception as e:
        print("Error in train_evaluate_classifier:", e)
        raise

def train_evaluate_multi_classifier(X_train, y_train, X_test, y_test, output_csv):
    classifiers = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3, learning_rate=0.1, n_estimators=100),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3),
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs')),
        'LinearSVM': make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=5)),
        'RBFSVM': SVC(kernel='rbf', probability=True),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    }

    results = []

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        num_classes = len(np.unique(y_train))

        if num_classes == 2:
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
        else:
            y_prob = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            tn, fp, fn, tp = [np.nan]*4
            specificity, sensitivity = np.nan, np.nan

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'algorithm': name,
            'train_count': len(y_train),
            'test_count': len(y_test),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'specificity': specificity,
            'sensitivity_recall': sensitivity,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'mse': mse,
            'mae': mae
        })

        print(f"{name} accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"All results saved to {output_csv}")

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap="Blues"):
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.show()
    except Exception as e:
        print("Error in plot_confusion_matrix:", e)
        raise


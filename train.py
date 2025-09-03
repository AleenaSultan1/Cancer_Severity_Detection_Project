#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from data_prep import load_and_prepare
from models import TorchNNClassifier


def run_training(filepath):
    # -------------------------------
    # Load data
    # -------------------------------
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(filepath)

    # -------------------------------
    # Sklearn models
    # -------------------------------
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
    }

    model_comparison = pd.DataFrame(columns=['Model', 'Train Time (s)', 'Test Accuracy',
                                             'Precision (Weighted)', 'Recall (Weighted)', 'F1 (Weighted)'])

    for name, clf in classifiers.items():
        print(f"\n----- Training {name} -----")

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        new_row = pd.DataFrame({
            'Model': [name],
            'Train Time (s)': [train_time],
            'Test Accuracy': [acc],
            'Precision (Weighted)': [report['weighted avg']['precision']],
            'Recall (Weighted)': [report['weighted avg']['recall']],
            'F1 (Weighted)': [report['weighted avg']['f1-score']]
        })
        model_comparison = pd.concat([model_comparison, new_row], ignore_index=True)

        print(f"Training time: {train_time:.2f} seconds")
        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'])
        plt.title(f"{name} Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    # -------------------------------
    # PyTorch Neural Network
    # -------------------------------
    print("\n----- Training Neural Network -----")
    X_train_nn = preprocessor.fit_transform(X_train)
    X_test_nn = preprocessor.transform(X_test)

    torch_clf = TorchNNClassifier(
        input_dim=X_train_nn.shape[1],
        hidden_dim=64,
        epochs=10,
        dropout_rate=0.2,
        random_state=42
    )

    start_time = time.time()
    torch_clf.fit(X_train_nn, y_train)
    nn_train_time = time.time() - start_time

    y_pred_nn = torch_clf.predict(X_test_nn)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    report_nn = classification_report(y_test, y_pred_nn, output_dict=True)

    new_row = pd.DataFrame({
        'Model': ['NeuralNetwork'],
        'Train Time (s)': [nn_train_time],
        'Test Accuracy': [acc_nn],
        'Precision (Weighted)': [report_nn['weighted avg']['precision']],
        'Recall (Weighted)': [report_nn['weighted avg']['recall']],
        'F1 (Weighted)': [report_nn['weighted avg']['f1-score']]
    })
    model_comparison = pd.concat([model_comparison, new_row], ignore_index=True)

    print(f"\nTraining time: {nn_train_time:.2f} seconds")
    print(f"Test Accuracy: {acc_nn:.4f}")
    print(classification_report(y_test, y_pred_nn))

    cm_nn = confusion_matrix(y_test, y_pred_nn)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title("Neural Network Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # -------------------------------
    # Final model comparison
    # -------------------------------
    print("\n=== Final Model Comparison ===")
    print(model_comparison.round(4))

    plt.figure(figsize=(12, 6))
    metrics = ['Test Accuracy', 'Train Time (s)', 'F1 (Weighted)']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        sns.barplot(x='Model', y=metric, data=model_comparison)
        plt.title(metric)
        plt.xticks(rotation=45)
        if metric == 'Train Time (s)':
            plt.yscale('log')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = os.path.join("data", "global_cancer_patients_2015_2024.csv")
    run_training(filepath)

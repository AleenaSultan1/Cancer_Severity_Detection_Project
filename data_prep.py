import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def compute_quantile_bins(y_values):
    """Split data into 3 equal-sized groups using quantiles"""
    q1 = np.quantile(y_values, 1 / 3)
    q2 = np.quantile(y_values, 2 / 3)
    return q1, q2


def bin_severity(score, q1, q2):
    """Assign class based on quantile thresholds"""
    if score <= q1:
        return 0
    elif score <= q2:
        return 1
    return 2


def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    features = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use',
                'Smoking', 'Obesity_Level', 'Treatment_Cost_USD']
    categorical_features = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

    # Convert continuous target to 3 classes
    q1, q2 = compute_quantile_bins(df['Target_Severity_Score'])
    df['Severity_Class'] = df['Target_Severity_Score'].apply(lambda x: bin_severity(x, q1, q2))

    print("\nClass distribution:")
    print(df['Severity_Class'].value_counts(normalize=True))

    X = df[features + categorical_features]
    y = df['Severity_Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor

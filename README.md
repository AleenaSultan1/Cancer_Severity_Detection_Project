Cancer Severity Classification Model

This project was developed as part of ECEG 478: Machine Learning. The goal was to design and evaluate machine learning models that can predict the severity of a cancer diagnosis using global patient data collected between 2015–2024.

The task is framed as a supervised learning classification problem, where models learn from labeled data to predict a Severity Score — a composite metric that reflects the seriousness of a patient’s cancer.

Four machine learning models were implemented and compared:

Logistic Regression

Random Forest

Gradient Boosting

Neural Networks

📊 Dataset Overview

The dataset contains patient-level features and a target severity score. Key features include:

Patient ID – Unique anonymized identifier.

Year – Year of diagnosis.

Age – Patient’s age (20–90).

Gender – Male, Female, or Other.

Country/Region – Geographic origin (captures regional disparities).

Cancer Type – Type of cancer (e.g., Breast, Lung, Colon).

Cancer Stage – Ordinal stage from 0 (in situ) to IV (advanced/metastatic).

Risk Factors – Composite variables such as:

Genetic predisposition

Air pollution exposure

Alcohol use

Smoking

Obesity

Treatment Cost – Estimated treatment cost (USD).

Survival Years – Years survived post-diagnosis.

Severity Score (Target Label) – Composite ordinal score (Low, Medium, High) based on cancer stage, risk factors, and survival outlook.

⚙️ Results
Random Forest

Training Time: 3.44s

Test Accuracy: 90.28%

Confusion Matrix & Heatmap:
<img width="519" height="391" alt="Random Forest Heatmap" src="https://github.com/user-attachments/assets/16a68868-d138-4aa1-8b9d-17a56a1949d6" />

Logistic Regression

Training Time: 0.20s

Test Accuracy: 99.81%

Confusion Matrix & Heatmap:
<img width="519" height="391" alt="Logistic Regression Heatmap" src="https://github.com/user-attachments/assets/2eecd3d0-6027-49ac-8c98-a4165fd16e83" />

Gradient Boosting

Training Time: 16.01s

Test Accuracy: 92.42%

Confusion Matrix & Heatmap:
<img width="519" height="391" alt="Gradient Boosting Heatmap" src="https://github.com/user-attachments/assets/22d52d1d-f98e-49de-be52-d3b30a675ca2" />

Neural Network

Training Time: 2.64s

Test Accuracy: 98.73%

Heatmap:
<img width="519" height="391" alt="Neural Network Heatmap" src="https://github.com/user-attachments/assets/449f1e38-5225-4084-977b-c9930e50d653" />

📈 Model Comparison

Comparison of training time, test accuracy, and F1 score across all models:

<img width="1184" height="590" alt="Model Comparison Chart" src="https://github.com/user-attachments/assets/aa01125b-33b8-4263-aea0-99e7b5014995" />
🚀 Key Insights

Logistic Regression surprisingly outperformed other models, achieving near-perfect accuracy.

Neural Networks also showed very high accuracy, with slightly higher computational cost.

Random Forest and Gradient Boosting performed reasonably well but were less accurate compared to Logistic Regression and Neural Networks.

Trade-off: Logistic Regression offered the best balance of accuracy and training efficiency.

# Cancer Severity Classification Model

This project was part of my coursework for ECEG 478: Machine Learning. The objective of this project is to build a complex neural network that can predict the severity of a cancer diagnosis using global patient data from 2015 to 2024. The type of machine learning implemented is supervised learning classification models, where the system learns patterns in labeled data to predict the Severity Score — a composite metric indicating how severe a patient’s cancer is. Through the project I explored four different machine learning models - including Logistic Regression, Neural Networks, Random Forests, and Gradient Boosting classifiers - and compared their performance.

#### Data Description
The data contains the following labels and target label:
● Patient ID: Each datapoint was assigned a unique patient ID number to ensure patient anonymity.
● Year: The year the cancer diagnosis was reported.
● Age: A continuous numerical variable representing the patient's age at the time of diagnosis. Values range from 20 to 90 years. This feature is essential as age is often correlated with both cancer risk and prognosis.
● Gender: A categorical variable with three possible values — Male, Female, or Other. Gender can influence cancer incidence and outcomes due to biological, behavioral, and healthcare access differences.
● Country/Region: A nominal categorical variable indicating the patient’s country or region of residence. This feature captures geographic disparities in cancer prevalence, environmental exposure, healthcare infrastructure, and survival rates.
● Cancer Type: A nominal categorical variable specifying the diagnosed type of cancer, such as Breast, Lung, Colon, etc. Cancer type significantly impacts the severity, treatment strategy, and expected survival outcomes.
● Cancer Stage: An ordinal categorical variable indicating the progression of the cancer at diagnosis. Stages range from Stage 0 (pre-cancer or in situ) to Stage IV (advanced/metastatic cancer). This is a critical predictor of prognosis and treatment complexity.
● Risk Factors: A composite set of binary or numerical features representing exposure to known cancer risk factors. These include:
● Genetic Risk (e.g., family history)
● Air Pollution Exposure
● Alcohol Use
● Smoking
● Obesity
These variables help estimate the likelihood of developing cancer and may correlate with severity or survival.
● Treatment Cost: A continuous numerical variable measuring the estimated cost of treatment in U.S. dollars (USD). Treatment cost may serve as a proxy for treatment intensity, resource allocation, or healthcare access level.
● Survival Years: A continuous numerical feature indicating the number of years the patient has survived post-diagnosis. This can be used to assess treatment efficacy or long-term prognosis. It's also potentially a target variable for survival analysis.
● Severity Score: The target label for prediction — a composite ordinal score representing the severity of the cancer diagnosis. It may be derived from a combination of cancer stage, risk factors, and survival outlook. The label was initially provided as a finite, continuous numeric value that I categorized into discrete classes (e.g., Low, Medium, High severity), this score guides triage and clinical prioritization.


#### Results 

###### Model Performance Results 

Random Forest Heat Map 
<img width="519" height="391" alt="image" src="https://github.com/user-attachments/assets/16a68868-d138-4aa1-8b9d-17a56a1949d6" />

Random Forest Model Confusion Matrix: 

----- Training RandomForest ----- 
Training time: 3.44 seconds 
Test Accuracy: 0.9028 
            precision recall f1-score support
            0.93       0.92       0.93   3368 
            0.84       0.87       0.86   3382 
            0.94       0.91       0.92   3250
accuracy                          0.90   10000
macro avg   0.90       0.90       0.90   10000 
weighted avg 0.90       0.90       0.90   10000

Logistic Regression Confusion Matrix
----- Training LogisticRegression ----- 
Training time: 0.20 seconds 
Test Accuracy: 0.9981 
            precision recall f1-score support
            1.00       1.00     1.00     3368 
            1.00       1.00     1.00     3382 
            1.00       1.00     1.00     3250
accuracy                        1.00     10000 
macro avg   1.00       1.00     1.00     10000 
weighted avg 1.00      1.00     1.00     10000

Logistic Regression Heat Map 
<img width="519" height="391" alt="image" src="https://github.com/user-attachments/assets/2eecd3d0-6027-49ac-8c98-a4165fd16e83" />

Gradient Boosting Confusion Matrix
----- Training GradientBoosting ----- 
Training time: 16.01 seconds 
Test Accuracy: 0.9242 
             precision recall f1-score support
             0.95        0.93     0.94    3368 
             0.88        0.90     0.89    3382 
             0.95 0.94 0.94 3250
accuracy                          0.92    10000 
macro avg    0.93 0.92 0.92 10000 
weighted avg 0.93 0.92 0.92 10000

Gradient Boosting Heat Map 
<img width="519" height="391" alt="image" src="https://github.com/user-attachments/assets/22d52d1d-f98e-49de-be52-d3b30a675ca2" />

Neural Network Heat Map 
<img width="519" height="391" alt="image" src="https://github.com/user-attachments/assets/449f1e38-5225-4084-977b-c9930e50d653" />

Training time: 2.64 seconds 
Test Accuracy: 0.9873 
             precision recall f1-score support
             0.99        0.99      0.99    3368 
             0.98        0.98      0.98    3382 
             0.99        0.99      0.99    3250
accuracy                           0.99    10000 
macro avg    0.99        0.99      0.99    10000 
weighted avg 0.99        0.99      0.99    10000

Model Comparison based on train time, test accuracy and F1 score.
<img width="1184" height="590" alt="image" src="https://github.com/user-attachments/assets/aa01125b-33b8-4263-aea0-99e7b5014995" />




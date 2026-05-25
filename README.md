# BigDataProject3: Cancer Diagnosis Using Machine Learning

## Description
This project uses Spark MLlib to build and compare two machine learning models for cancer diagnosis. The dataset contains 569 patient samples with 20 clinical features, labeled as 
B (Benign) or M (Malignant). Two algorithms, Random Forest and Gradient Boosting, are trained and evaluated using F1 score, precision, recall, and accuracy.

## Dataset
- Total Samples: 569 patient samples
- Features: 20 clinical variables
- Labels: B (Benign), M (Malignant)

## How to Run

### Requirements
- Python 3.10+
- Java 17+
- PySpark 4.1

### Install Dependencies
pip install pyspark

### Run
python3 main.py

## Results
                  Random Forest    Gradient Boosting
F1 Score              0.9535            0.9650
Precision             0.9535            0.9652
Recall                0.9535            0.9651
Accuracy              95.3488%          96.5116%
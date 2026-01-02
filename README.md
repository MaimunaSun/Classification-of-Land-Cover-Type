# Land Cover Classification: Model Comparison

This project classifies forest land cover types using **machine learning** and compares the performance of **Gaussian Naive Bayes (NB)**, **Multi-Layer Perceptron (MLP) Neural Network**, and **Support Vector Machine (SVM)** classifiers.

---

## Project Overview
- **Dataset:** 15,120 samples with 54 features (terrain, soil, and hydrology)  
- **Target:** `Cover_Type` (7 classes)  
- **Goal:** Predict the land cover type for given terrain and soil features using multiple models and compare their performance.

---

## Workflow

### 1. Data Preprocessing
- Removed the `Id` column  
- Split dataset into **features** and **target**  
- Divided data into training (80%) and testing (20%) sets  

### 2. Modeling

#### Gaussian Naive Bayes
- Simple probabilistic classifier based on **Bayes’ theorem**  
- **Accuracy:** ~61%  
- **Strengths:** Fast and easy to implement  
- **Weaknesses:** Struggles with correlated features  

#### MLP Neural Network
- Deep neural network with hidden layers `(150, 100, 50, 20)`  
- Activation: ReLU, Solver: Adam  
- **Accuracy:** ~81%  
- **Strengths:** Handles complex patterns and feature interactions  
- **Weaknesses:** Slower to train, requires more computation  

#### Support Vector Machine (SVM)
- Linear kernel classifier for **binary/multi-class separation**  
- **Accuracy:** ~71%  
- **Strengths:** Effective for linear boundaries, good recall for dominant classes  
- **Weaknesses:** Slower on large datasets, some minority class misclassification  

---

### 3. Evaluation
- Generated **classification reports** for each model  
- Plotted **confusion matrices** to visualize prediction accuracy per class  
- Observed class imbalances and misclassifications  

---

### 4. Prediction on Unseen Test Data
- Models used to predict `Cover_Type` for a separate test set  
- Compared **predicted class distributions**  
- Visualized frequencies using **bar charts**  

---

## Results
| Model | Accuracy | Notes |
|-------|---------|-------|
| Gaussian Naive Bayes | 61% | Fast but struggles with correlated features |
| MLP Neural Network | 81% | Handles complex interactions, best overall performance |
| SVM (linear) | 71% | Good linear separation, moderate accuracy, slower training |

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  

---

## Key Takeaways
- **MLP outperforms Naive Bayes and SVM** for this dataset  
- Class imbalance affects predictions; visualizations help understand the distribution  
- Demonstrates a clear case of **shallow vs deep learning vs SVM models** in practice


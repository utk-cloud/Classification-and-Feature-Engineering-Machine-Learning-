# Classification and Feature Engineering Project  

## Overview  
This project focuses on building and evaluating classification models using feature engineering techniques to predict the success of marketing campaigns. It also explores the use of pre-trained neural networks for feature extraction to enhance model performance.

---

## Objectives  

### **Part 1: Classification and Feature Engineering**
1. **Exploratory Data Analysis (EDA)**:  
   - Identify usable variables and their relationships.  
   - Assess class balance and define target classes.  

2. **Metric Selection**:  
   - Choose evaluation metrics such as accuracy, F1 score, balanced accuracy, and AUC, with reasoning.  

3. **Variable Filtering and Encoding**:  
   - Normalize or transform continuous variables where necessary.  
   - Encode or reduce discrete variable values for better generalization.  
   - Handle noisy or immaterial variables.  

4. **Test Data Preparation**:  
   - Create balanced and representative test datasets.  

5. **Model Training**:  
   - Train and tune the following models using cross-validation:  
     - SVM with RBF kernel.  
     - Neural network with ReLU hidden layer and Softmax output.  
     - Random forest.  

6. **Feature Importance Analysis**:  
   - Determine significant features for each model and analyze overlaps.  

7. **Feature Elimination**:  
   - Experiment with recursive feature elimination to improve model performance.  

8. **Model Evaluation**:  
   - Evaluate promising models on test datasets and assess their business utility.  

9. **Pathological Testing**:  
   - Test model performance under specific data splits, such as temporal or demographic segregation.  

### **Part 2: Pre-trained Neural Networks for Feature Extraction**
10. **Transfer Learning**:  
    - Use ResNet18 as a fixed feature extractor with PyTorch.  

11. **Feature Extraction**:  
    - Implement a function to extract ResNet18 features for training data.  

12. **Model Comparison**:  
    - Compare the performance of L2-regularized logistic regression and random forest on extracted features.  

13. **Findings and Conclusions**:  
    - Summarize insights from experiments and model evaluations.  

---

## Dataset  
The project uses the **Bank Marketing Dataset**, which contains historical data from marketing campaigns, including customer demographics, call details, and campaign outcomes.  
**Source**: [Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)  

---

## Tools and Libraries  
- **Python 3.8+**  
- **Libraries**:  
  - Pandas, NumPy, Matplotlib, Seaborn (EDA and preprocessing)  
  - Scikit-learn (model training, feature selection, and evaluation)  
  - PyTorch (transfer learning and feature extraction)  

---

## Evaluation Metrics  
- **Accuracy**  
- **F1 Score**  
- **Area Under the Curve (AUC)**  
- **Balanced Accuracy**  

---


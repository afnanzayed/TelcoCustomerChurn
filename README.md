# Telecom Customer Churn Prediction

## Project Overview
    This project focuses on predicting customer churn in the telecom sector, which is critical for reducing revenue loss and improving customer loyalty. By identifying potential churners, telecom companies can take proactive steps to enhance customer retention. This project leverages multiple machine learning models to gain insights and provide an effective predictive solution.## Project Structure

1. **Data Exploration**
   Initial examination of the dataset, including:
   - Checking for missing values
   - Understanding categorical vs. numerical features
   - Basic statistical summary

2. **Data Transformation**
   Standardizing numerical features to ensure consistent scaling, which enhances model performance.

3. **Exploratory Data Analysis (EDA)**
   Detailed analysis to uncover patterns and correlations in the data, which helps to identify important features related to churn.

4. **Model Building**
   Steps taken to prepare the data for machine learning, including:
   - **Preparing Data for Modeling**: Splitting data into training and testing sets.
   - **Standardizing Numeric Attributes**: Scaling numerical columns for consistent model performance.

5. **Models**
  Multiple machine learning models were tested, tuned, and evaluated to ensure robust prediction accuracy and identify the most effective approach.
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Classifier (SVC)**
   - **Random Forest Classifier**
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **AdaBoost Classifier**
   - **Gradient Boosting Classifier**
   - **Voting Classifier**
   - **XGBoost Classifier**

6. **Model Evaluation**
   Each model is evaluated using metrics such as accuracy, confusion matrix, and ROC-AUC to determine the best-performing model.

## Results and Insights

   - **High Churn Risk Factors**: Customers with month-to-month contracts, electronic check payments, and fiber optic internet service are more likely to churn.
   - **Demographic Insights**: Churn rates are higher among customers without dependents or partners, non-senior citizens, and those lacking online security or tech support.
   - **Billing and Service Preferences**: Customers using paperless billing and phone service show distinct churn patterns.
      Business Impact: These insights highlight specific customer segments for targeted retention strategies, enabling telecom companies to address churn factors more effectively.

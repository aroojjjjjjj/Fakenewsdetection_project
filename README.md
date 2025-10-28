# Fakenewsdetection_project
This repository contains weekly assignments for my Data Science course project.  
The project is assigned based on roll number, and my task is **Fake News Detection**.

📅 Weekly Progress

 ✅ Week 1: Orientation & Setup
- Installed Python (Google Colab used).  
- Created GitHub repository for project.  
- Downloaded sample dataset (Titanic/Iris) and displayed first 10 rows.  

✅ Week 2: Data Collection & Cleaning
- Collected Fake News dataset.  
- Performed data cleaning:
  - Removed rows with missing values.  
  - Verified dataset contains no duplicates.  
- Saved and uploaded cleaned dataset to GitHub.  

 ✅ Week 3: Exploratory Data Analysis (EDA)
- Created 5 different visualizations using Matplotlib & Seaborn:
  1. News Articles per Category (bar chart)  
  2. Fake vs Real News Distribution (bar chart)  
  3. Distribution of Article Lengths (histogram)  
  4. Article Length vs Category (scatter plot)  
  5. Top 10 Authors by Articles (bar chart)  
- Added short insights under each plot.

✅ Week 4: Statistics & Probability
-Calculated mean, median, mode, and variance of dataset features.
-Performed correlation analysis between numeric features and target variable (label).
-Found the 3 most related features to the target:
1.text_length
2.title_length
3.One categorical feature (weak relation but considered).

✅ Week 5: Supervised Learning – Regression
- Trained a baseline Linear Regression model using `text_length` as predictor.  
- Performed train/test split (80/20).  
- Evaluated model performance:  
  - Mean Absolute Error (MAE): ~0.50  
  - Root Mean Squared Error (RMSE): ~0.50  
- Learned how regression can be applied for binary target variables (baseline). 

 ✅ Week 6: Supervised Learning – Classification
-Implemented two classification models:
1.Logistic Regression → Baseline classifier
2.Random Forest Classifier → Ensemble method
-Compared their accuracy on the dataset.
-Identified which model performs better for detecting Fake vs Real news.

 ✅ Week 7: Model Evaluation
.Evaluated trained classification models using key performance metrics: Precision, Recall, F1 Score, and Accuracy.
.Generated and analyzed the Confusion Matrix and ROC Curve for better understanding of model performance.
.Observed that the model achieved high accuracy (≈100%) on the test set.
.Identified F1 Score as the most important metric for the Fake News Detection project, since it balances precision and recall  both critical for minimizing false predictions.
.Finalized F1 Score as the main evaluation metric for upcoming project stages.

 📌 Project Milestones So Far
- Environment setup complete.  
- Dataset collected and cleaned.  
- Initial exploratory data analysis completed.  
- Identified the key predictive variables most related to the target variable (label).
- Built first baseline regression model.
- Built first baseline classification models and compared accuracy.
- Finalized F1 Score as the primary evaluation metric for the Fake News Detection model.

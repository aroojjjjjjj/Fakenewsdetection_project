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

✅ Week 8: Unsupervised Learning
. Implemented K-Means clustering to explore hidden patterns in the dataset.
. Used TF-IDF vectorization to convert text data into numerical features.
. Applied PCA (Principal Component Analysis) to visualize clusters in 2D space.
. Observed that Cluster 1 contained mostly fake news while Cluster 0 had more real news, showing a natural separation even      without labels.
. Demonstrated how unsupervised learning can uncover structure and relationships in textual data.

✅ Week 9: Neural Networks Basics
.Implemented a simple Artificial Neural Network (ANN) using Keras.
.Used text length as a numerical feature to train the neural model.
.Trained the ANN with multiple epochs to learn patterns in fake vs real news.
.Evaluated ANN performance on test data and recorded accuracy.
.Compared ANN results with previous models (Logistic Regression and Random Forest).
.Confirmed that ANN can learn non-linear patterns and serves as a strong neural baseline for the project.

✅ Week 10: Advanced Deep Learning (CNN & RNN)
.Implemented a Convolutional Neural Network (CNN) using news‑related images to classify fake vs real news.
.Organized image data into fake and real classes and trained the CNN model for image‑based prediction.
.Implemented a Recurrent Neural Network (RNN) using textual news data after tokenization and sequence padding.
.Trained the RNN model on news article text to learn sequential patterns in fake vs real news.
.Generated predictions using both CNN (image data) and RNN (text data).
.Confirmed the effectiveness of specialized deep learning models for handling different data modalities (images and text).

✅ Week 11: Natural Language Processing (NLP)
. Performed NLP preprocessing on news text including tokenization and stopword removal.
. Converted text data into numerical features using TF-IDF vectorization.
. Split dataset into training and testing sets.
. Trained a machine learning classifier on TF-IDF features.
. Evaluated model using accuracy, precision, recall, and F1-score.
. Established a complete NLP pipeline for fake news text classification.

✅ Week 12: report
📄 2‑page industry application report
🏥 Healthcare + 💰 Finance case studies included

✅ Week 13: Model Deployment
Saved the trained Fake News Detection model and TF‑IDF vectorizer using pickle.
Implemented a Flask application to load the saved model for inference.
Created API routes to accept text input and return prediction results (REAL / FAKE).
Successfully ran the Flask app on localhost/Colab environment.
Tested the deployed pipeline by passing sample news text and receiving correct predictions.
Demonstrated complete end‑to‑end workflow from input text to model prediction.

✅ Week 14: Ethics & Explainability
Studied ethical considerations in AI systems, including fairness, transparency, and accountability.
Reviewed interpretable machine learning concepts using reference materials by Christoph Molnar and Google AI Ethics.
Applied SHAP to explain predictions of the fake news detection model.
Visualized how individual words contributed to REAL or FAKE predictions.
Improved model transparency by explaining why specific inputs led to specific outputs.
Ensured the AI system is interpretable and ethically aligned for real‑world usage.


 📌 Project Milestones So Far
- Environment setup complete.  
- Dataset collected and cleaned.  
- Initial exploratory data analysis completed.  
- Identified the key predictive variables most related to the target variable (label).
- Built first baseline regression model.
- Built first baseline classification models and compared accuracy.
- Finalized F1 Score as the primary evaluation metric for the Fake News Detection model.
- Added unsupervised analysis (K-Means clustering with PCA visualization) to the Fake News Detection project.
- ANN baseline model successfully implemented and evaluated against traditional classifiers.
- Applied specialized AI models by implementing CNN for image data and RNN for text‑based fake news detection.
- NLP preprocessing and feature extraction pipeline successfully completed.
- Project successfully connected to real‑world industry applications and case studies.
- End‑to‑end fake news detection pipeline successfully deployed and tested using Flask.
- Added explainability to the fake news detection model by interpreting predictions using SHAP.

  

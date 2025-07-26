# Iris Dataset Exploration and Classification

## Description

This project focuses on the **exploration and classification** of the well-known Iris dataset. It includes detailed visualizations and statistical summaries to understand the structure of the data. In addition, a machine learning model is built to classify iris species based on four input features.

The project also includes a simple **web-based application** that runs locally on your computer. This application allows users to enter iris measurements and predict the flower class in real time using the trained model.

---

## Key Features

- Exploratory Data Analysis (EDA) using plots such as histograms, boxplots, and pairplots.
- Descriptive statistics to summarize feature distributions.
- Training and evaluation of a classification model (e.g., Logistic Regression,VotingClassifier,StackingClassifier,KNeighborsClassifier,Decision Tree Classifier the final Model is the Votting Classifier with base learners Support Vector Machine,Logistic Regression and Decision Tree Classifier).
- A local web application for interactive prediction of flower species.

---
## Project Overview

The Iris data set is one of the most well-known datasets in data science and machine learning. It includes 150 samples from three species of Iris (setosa, versicolor, and virginica), with four features recorded for each sample:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

This project performs the following tasks:
- Data loading and inspection
- Summary statistics and distribution analysis
- Data visualization using seaborn and matplotlib
- Correlation analysis and boxplots
- Class-wise comparisons

## Contents

- `Iris Data Set Exploration.ipynb`: Jupyter notebook with complete code and output
- `iris_scaler_voting_clf.pkl`: Pickle File Containing Model to Scale the Data before passing it to Actual Model
- `iris_voting_clf.pkl`: Actual Model
- `README.md`: Description of the project and usage instructions
- `Iris.csv`: CVS File Contining data
- `app.py`: File containing streamlit code of Deployment

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (To predict Class)

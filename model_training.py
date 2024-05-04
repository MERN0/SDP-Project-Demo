import numpy as np
import pandas as pd
# model selection
# split data into training and test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # preformance of the model

# loading dataset to pandas dataframe
credit_card_data = pd.read_csv('data_files/creditcard_2023.csv')

# first 5 rows of the dataset
credit_card_data.head()

# dataset information
credit_card_data.info()

# checking the number of missing values in each column
credit_card_data.isnull().sum()

# distribution of legit transactions & fraudulent transactions
# 0 = transactions
# 1 = fraud transactions
credit_card_data['Class'].value_counts()

"""This Dataset is highly unblanced

0 --> Normal Transaction

1 --> fraudulent transaction

"""

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measures of the data
# amount describes the amount of money in each transaction
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions
# this is how we prove a transaction is legit or fradulant
# by compairing the mean of amount
# mean values of all the columns
credit_card_data.groupby('Class').mean()

"""Under-Sampling

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

Number of Fraudulent Transactions --> 242
"""

legit_sample = legit.sample(n=242)

"""
Concatenating two DataFrames"""

new_dataset = pd.concat([legit_sample, fraud], axis=0)
# axis 0 adds dataset 1 by one
# 1 adds column wise

new_dataset.head() # 0 is present

new_dataset.tail() # 1 is present

new_dataset['Class'].value_counts() # same number of legit and fraud transactions

new_dataset.groupby('Class').mean()

"""Splitting the data into Features & Targets"""

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X) # we dropped the class column previously there were 31 columns

print(Y) # contains labels 0 and 1

"""Split the data into Training data & Testing Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)
# stratify = y gives an even distribution of 0 and 1 for training and testing data

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Logistic Regression for binary classification
"""

model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

# A = new_dataset.drop(columns='Class', axis=1)
# B = new_dataset['Class']

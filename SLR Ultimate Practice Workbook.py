# Ultimate Practice Workbook: Simple Linear Regression

# This workbook is designed to solidify the systemic flow of simple linear regression,
# ensuring complete mastery of data flow, mathematical foundations, and pipeline architecture.


# Phase 1: Foundational Logic & System Architecture

# Before touching any syntax, you must be able to whiteboard the data flow.
# The goal is to calculate a best-fit straight line (y = intercept + coefficient * x) that minimizes the residual errors.

# Load CSV data -> Extract features -> Split into train/test -> Reshape for model -> Fit model -> Evaluate on test set.

# Task 1.1: The Pipeline Map

# Ingestion: Load the raw CSV data into a structured format.
# Extraction: Isolate feature columns into NumPy arrays.
# Partitioning: Split the data mutually exclusively (e.g., 80% train / 20% test).
# Transformation: -
#   Reshape 2D features into 1D matrices for compatibility with scikit-learn's selected model - "train_test_split" by using the to_numpy() method.
#   Reshape 1D features into 2D matrices for compatibility with scikit-learn's selected model - "LinearRegression" by using the reshape(-1, 1) method.
# Execution: Pass the training data through the model to find the optimal coefficients.
# Evaluation: Calculate MSE and R2-Score on the unseen test set.


# Phase 2: Environment & Data Ingestion

# Setting up a clean local Python virtual environment in your workspace is crucial for keeping dependencies isolated before beginning any analysis.

# Task 2.1: Library Installation and Import

# pip install numpy pandas scikit-learn matplotlib python-dotenv

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the hidden variables from the .env file

load_dotenv()

# Your saved URL or Data set path should be stored in the .env file as a variable named "dataset_path"

dataset_path = os.getenv("dataset_path")

# Task 2.2: Data Loading

# url = "link" or path = "local_path_to_csv"
# df = pd.read_csv(url/path)

df = pd.read_csv(dataset_path)
print(df.sample(5))  # Display a random sample of 5 rows to verify successful ingestion
print(df.describe())  # Display basic statistics to understand the data distribution


# Phase 3: Extraction & Visualization (Avoiding Cognitive Debt)

# Blindly feeding data into an algorithm is a core anti-pattern.
# You must visualize the relationship first to confirm if a linear model is even appropriate.

# Task 3.1: Feature Extraction
# cdf = df[['Column Header/s']]

cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

# Display a random sample of 5 rows to verify successful extraction
print(cdf.sample(5))

# Task 3.2: Visualize Feature Relationships

# Use histograms to visualize the distributions of the features and target variable.

cdf.hist(figsize=(10, 8))
plt.show()

# As you can see, most engines have 4, 6, or 8 cylinders, and engine sizes between 2 and 4 liters.
# As you might expect, combined fuel consumption and CO2 emission have very similar distributions.
# Go ahead and display some scatter plots of these features against the CO2 emissions, to see how linear their relationships are.

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# This is an informative result.
# Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions.
# Their intercepts are similar, while they noticeably differ in their slopes.

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0, 27)
plt.show()

# Although the relationship between engine size and CO2 emission is quite linear,
# you can see that their correlation is weaker than that for each of the three fuel consumption groups.
# Notice that the x-axis range has been expanded to make the two plots more comparable.

# Practice exercise 1
# Plot __CYLINDER__ against CO2 Emission, to see how linear their relationship is.
# Type your code below and run it to see the result.

# Task 3.3: # Extract the input feature and labels from the dataset.

# Although perhaps not necessarily the ideal choice of input feature,
# for illustration purposes, you will use engine size to predict CO2 emission with a linear regression model.
# You can begin the process by extracting the input feature and target output variables, X and y, from the dataset.
# X = cdf.Column Header/s.to_numpy()
# y = cdf.Column Header/s.to_numpy()

X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()


# Phase 4: The Core Pipeline (Train/Test Split & Reshaping)

# Task 4.1: Create train and test datasets

# Next, you will split the dataset into mutually exclusive training and testing sets.
# You will train a simple linear regression model on the training set and estimate its ability to-
# -generalize to unseen data by using it to make predictions on the unseen testing data.

# Since the outcome of each data point is part of the testing data,
# you have a means of evaluating the out-of-sample accuracy of your model.

# Now, you want to randomly split your data into train and test sets,
# using 80% of the dataset for training and reserving the remaining 20% for testing.

# Which fraction to use here mostly depends on the size of your data,
# but typical testing sizes range from 20% to 30%.

# The smaller your data, the larger your training set needs to be because it's easier to find spurious patterns in smaller data.
# The downside is that your evaluation of generalizability will have less reliability.
# Bigger is better when it comes to data.

# use the train_test_split function from scikit-learn to split the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# The outputs are one-dimensional NumPy arrays or vectors.

print(type(X_train), np.shape(X_train), np.shape(y_train))

# Task 4.2: fit the model to the training data

# Scikit-learn expects 2D arrays.
# Use the LinearRegression class from scikit-learn to create a linear regression model object.

# Create a model object
regressor = linear_model.LinearRegression()

# Train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it.
# We can let it infer the number of observations using '-1'.
# Use the fit method to train the model on the training data.

regressor.fit(X_train.reshape(-1, 1), y_train)

# Task 4.3: Evaluate the model

# Print the coefficients

print("Coefficients: ", regressor.coef_[0])
# with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print("Intercept: ", regressor.intercept_)

# Here, __Coefficient__ and __Intercept__ are the regression parameters determined by the model.
# They define the slope and intercept of the 'best-fit' line to the training data.

# Task 4.4: Intentional Debugging (Shape Mismatch)

# # INTENTIONAL ERROR: Passing 1D array
# # Run this exact code in your terminal or script to trigger a ValueError,
# # then read the traceback to deeply understand the matrix dimension failure.
# regressor.fit(X_train, y_train)

# Task 4.5: The Architectural Fix

# Observe how dynamic reshaping resolves the dimension issue.

# # Correct implementation
# regressor.fit(X_train.reshape(-1, 1), y_train)

# Task 4.6: Visualize model outputs

# You can visualize the goodness-of-fit of the model to the training data by plotting the fitted line over the data.
# The regression model is the line given by y = intercept + coefficient * x.

plt.scatter(X_train, y_train, color="blue")
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")


# Phase 5: Model Training & Evaluation

# Compare your model's predictions against actual labels mathematically.
# Metric
# Definition
# Code Implementation

# * Mean Absolute Error (MAE)
# Average absolute error between predictions and actuals.
# mean_absolute_error(y_test, y_pred)

# * Mean Squared Error (MSE)
# Mean of the squared error (penalizes larger errors heavily).
# mean_squared_error(y_test, y_pred)

# * R2-Score
# Goodness of fit metric (a perfect model scores 1.0).
# r2_score(y_test, y_pred)

# Model evaluation
# You can compare the actual values and predicted values to calculate the accuracy of a regression model.
# Evaluation metrics play a key role in the development of a model,
# as they provide insight into areas that require improvement.

# There are different model evaluation metrics:

# * Mean Absolute Error: It is the mean of the absolute value of the errors.
# This is the easiest of the metrics to understand since it’s just an average error.

# * Mean Squared Error (MSE): MSE is the mean of the squared error.
# In fact, it's the metric used by the model to find the best fit line,
# and for that reason, it is also called the residual sum of squares.

# * Root Mean Squared Error (RMSE).
# RMSE simply transforms the MSE into the same units as the variables being compared, which can make it easier to interpret.

# * R2-Score is not an error but rather a popular metric used to estimate the performance of your regression model.
# It represents how close the data points are to the fitted regression line.
# The higher the R2-Score value, the better the model fits your data.
# The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

# Use the predict method to make test predictions
y_pred = regressor.predict(X_test.reshape(-1, 1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))


# Phase 6: Advanced Implementation (From Scratch)

# I have uploaded two PDF files to the repository for manual calculation using pen and paper.
# The PDFs contain the mathematical derivation of the slope and intercept for a simple linear regression model.
# I recommend you read through them and practice on a piece of paper to understand the underlying math before attempting to implement it in Python.

# Task 6.1: Manual Calculation Challenge

# Before relying heavily on the scikit-learn black box,
# write a raw Python function without any machine learning libraries.
# You must calculate the mean of your X and y arrays,
# compute the variance and co-variance, and
# manually derive the slope and intercept.

# Step 1: Calculate the means

# # Sample arrays (Engine Size vs CO2 Emissions)
# X = np.array([2.0, 2.4, 1.5, 3.5, 4.0])
# y = np.array([196, 221, 136, 255, 290])

# or

Mean_X_array = np.mean(X_train)
Mean_y_array = np.mean(y_train)
print("Mean of X_array:%.2f" % Mean_X_array)
print("Mean of y_array:%.2f" % Mean_y_array)

# Step 2: Calculate Variance and Covariance
# Variance: sum of (x - Mean_X_array)^2
# We are using the training data to calculate the variance and covariance, as this is what the model would have learned from.

# Variance measures how far a set of numbers are spread out from their average value.
# First we minus the mean from each value in the array,
# then square the result, and finally sum all of those squared differences together to get the variance.

variance_x = np.sum((X_train - Mean_X_array) ** 2)

# Covariance: sum of (x - Mean_X_array) * (y - Mean_y_array)
# Covariance measures how much two random variables vary together.
# First we minus the mean from each value in the X and y arrays,
# then multiply the results together, and finally sum all of those products together to get the covariance

covariance_xy = np.sum((X_train - Mean_X_array) * (y_train - Mean_y_array))

# Step 3: Derive slope (m) and intercept (c)

# Derive the slope (m) measures the steepness of the line
m = covariance_xy / variance_x

# Derive the intercept (c) measures where the line crosses the y-axis.
c = Mean_y_array - (m * Mean_X_array)

print(f"Calculated Slope (m): {m}")
print(f"Calculated Intercept (c): {c}")


# Phase 7: Automated Testing

# Task 7.1: Pipeline Assertions

# Write assertion tests to validate your data partitioning logic mathematically.

# # Test that no data was lost during the split
# assert len(X_train) + len(X_test) == len(X), "Data leakage or loss detected during split!"

# The "Why" Behind Automated Assertions

# Think of an `assert` statement like a strict, automated grading rubric.
# When introducing Python basics to a new batch of students, you establish absolute rules that cannot be broken.
# The `assert` keyword does exactly this for your system architecture: it is a tripwire.

# In machine learning, data leakage or dropped rows during the Train/Test split will silently destroy your model's credibility.
# The algorithm won't crash; it will just train on bad math and give you confident, incorrect answers.
# An `assert` statement mathematically guarantees the integrity of your data flow before the model is even allowed to train.


# Step-by-Step Breakdown of Task 7.1

# **The Goal:**
# We need to prove mathematically that when we split a dataset into an 80% training chunk and a 20% testing chunk,
# no rows were lost or duplicated in the process.
# The sum of the parts must equal the whole.

# **The Syntax:**
# An assertion has two parts: `assert [The Absolute Truth], "[The Error Message if it's a Lie]"`

# 1. **The Absolute Truth:** `len(X_train) + len(X_test) == len(X)`
# * We are asking Python to count the rows in the training set (`len(X_train)`) and add them to the rows in the testing set (`len(X_test)`).
# * We then use the equality operator (`==`) to verify if that sum matches the exact total row count of our original, unsplit dataset (`len(X)`).


# 2. **The Error Message:** `"Data leakage or loss detected during split!"`
# * If the math checks out (True), the `assert` statement does absolutely nothing. The script continues running silently.
# * If the math fails (False), the script violently halts, crashes the program, and prints this exact error message to your terminal.


# The Intentional Failure Exercise

# To truly understand how this protects your pipeline, you need to break it on purpose.
# Drop this script into your VS Code workspace and run it.

# 1. We start with a dataset of 10 engine sizes
X_Example = np.array([1.0, 1.5, 2.0, 2.4, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])

# 2. We split it properly (8 for training, 2 for testing)
X_train_Example = np.array([1.0, 1.5, 2.0, 2.4, 3.0, 3.5, 4.0, 4.5])
X_test_Example = np.array([5.0, 5.5])

# 3. The Tripwire (This will pass silently because 8 + 2 = 10)
assert len(X_train_Example) + len(X_test_Example) == len(
    X_Example
), "Data leakage or loss detected!"
print("Pipeline Check 1: Passed. No data lost.")

# 4. INTENTIONAL SABOTAGE
# We simulate a bug where one row of training data accidentally gets dropped
X_train_corrupted = np.array([1.0, 1.5, 2.0, 2.4, 3.0, 3.5, 4.0])  # Only 7 items now!

# 5. The Tripwire Triggered
print("\nRunning Pipeline Check 2...")
assert len(X_train_corrupted) + len(X_test_Example) == len(
    X_Example
), "CRITICAL ERROR: Data leakage or loss detected!"

# When you run this, you will see the first check pass, and the second check immediately trigger an `AssertionError`.
# This is how you build robust, self-testing data pipelines.

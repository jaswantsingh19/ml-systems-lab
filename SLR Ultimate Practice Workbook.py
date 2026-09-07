# Ultimate Practice Workbook: Simple Linear Regression

# This workbook is designed to solidify the systemic flow of simple linear regression,
# ensuring complete mastery of data flow, mathematical foundations, and pipeline architecture.


# Phase 1: Foundational Logic & System Architecture

# Before touching any syntax, you must be able to whiteboard the data flow.
# The goal is to calculate a best-fit straight line (y = intercept + coefficient * x) that minimizes the residual errors.

# Task 1.1: The Pipeline Map

# Ingestion: Load the raw CSV data into a structured format.
# Extraction: Isolate feature columns into NumPy arrays.
# Partitioning: Split the data mutually exclusively (e.g., 80% train / 20% test).
# Transformation: Reshape 1D features into 2D matrices.
# Execution: Pass the training data through the model to find the optimal coefficients.
# Evaluation: Calculate MSE and R2-Score on the unseen test set.

# Phase 2: Environment & Data Ingestion
# Setting up a clean local Python virtual environment in your workspace is crucial for keeping dependencies isolated before beginning any analysis.
# Task 2.1: Library Installation
# pip install numpy pandas scikit-learn matplotlib


# Task 2.2: Data Loading
# import pandas as pd
# url = "link" or path = "local_path_to_csv"
# df = pd.read_csv(url)


# Phase 3: Extraction & Visualization (Avoiding Cognitive Debt)

# Blindly feeding data into an algorithm is a core anti-pattern.
# You must visualize the relationship first to confirm if a linear model is even appropriate.

# Task 3.1: Feature Extraction

# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# X = cdf.ENGINESIZE.to_numpy()
# y = cdf.CO2EMISSIONS.to_numpy()


# Phase 4: The Core Pipeline (Train/Test Split & Reshaping)

# Task 4.1: Intentional Debugging (Shape Mismatch)

# Scikit-learn expects 2D arrays.
# Run this exact code in your terminal or script to trigger a ValueError,
# then read the traceback to deeply understand the matrix dimension failure.
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# # INTENTIONAL ERROR: Passing 1D array
# regressor.fit(X_train, y_train)


# Task 4.2: The Architectural Fix

# Observe how dynamic reshaping resolves the dimension issue.

# # Correct implementation
# regressor.fit(X_train.reshape(-1, 1), y_train)


# Phase 5: Model Training & Evaluation

# Compare your model's predictions against actual labels mathematically.
# Metric
# Definition
# Code Implementation

# Mean Absolute Error (MAE)
# Average absolute error between predictions and actuals.
# mean_absolute_error(y_test, y_pred)
# Mean Squared Error (MSE)
# Mean of the squared error (penalizes larger errors heavily).
# mean_squared_error(y_test, y_pred)
# R2-Score
# Goodness of fit metric (a perfect model scores 1.0).
# r2_score(y_test, y_pred)


# Phase 6: Advanced Implementation (From Scratch)

# Task 6.1: Manual Calculation Challenge

# Before relying heavily on the scikit-learn black box,
# write a raw Python function without any machine learning libraries.
# You must calculate the mean of your X and y arrays,
# compute the variance and covariance, and
# manually derive the slope and intercept.

# Phase 7: Automated Testing

# Task 7.1: Pipeline Assertions

# Write assertion tests to validate your data partitioning logic mathematically.

# # Test that no data was lost during the split
# assert len(X_train) + len(X_test) == len(X), "Data leakage or loss detected during split!"

import pandas as pd
import numpy as np

# Load data from file
try:
    df = pd.read_csv("hetero.txt", sep="\t")
except FileNotFoundError:
    print("File not found.")
    exit()
except pd.errors.EmptyDataError:
    print("File is empty.")
    exit()

# Check that all columns are present and have the correct data type
if set(df.columns) != {"study number", "rsid", "p value", "sample size"}:
    print("File format is incorrect.")
    exit()
if not df[["study number", "sample size"]].apply(pd.to_numeric, errors="coerce").notnull().all().all():
    print("Invalid numeric data.")
    exit()

# Transform p-value to -log10(p-value)
df["log_pvalue"] = -np.log10(df["p value"])

# Calculate mean and covariance matrix of log-transformed p-values
mean = df["log_pvalue"].mean()
cov = df[["log_pvalue", "sample size"]].cov()

# Calculate Mahalanobis distance for each data point
df["mahalanobis"] = df[["log_pvalue", "sample size"]].apply(lambda x: np.sqrt(np.dot(np.dot((x - [mean, x[1]]), np.linalg.inv(cov)), (x - [mean, x[1]]))), axis=1)

# Determine the threshold for outlier detection
threshold = np.mean(df["mahalanobis"]) + (3 * np.std(df["mahalanobis"]))

# Remove outliers
df = df[df["mahalanobis"] <= threshold]

# Save cleaned data to file
df.to_csv("cleaned_data.txt", sep="\t", index=False)


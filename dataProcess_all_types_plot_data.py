import pandas as pd
import glob
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    GlobalMaxPooling1D,
    AveragePooling1D,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
    TimeDistributed,
    LSTM,
)
from sklearn.model_selection import train_test_split


dir_path = os.path.dirname(os.path.realpath(__file__))
# Get a list of all CSV files in a directory
csv_files = glob.glob(dir_path + '/*.csv')

default_csv = glob.glob(dir_path +"/default" + '/*.csv')

print(default_csv)

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

axis_df = pd.read_csv(default_csv[0], header=None)
axis_df = axis_df[[8,9,10,11]]
print(axis_df)

def bin_value(value):
    if(value < 4):
        return 0
    elif(value < 8):
        return 1
    elif(value < 12):
        return 2
    elif(value < 16):
        return 3
    elif(value <= 20):
        return 4

samples = []
X_list = []
y_list = []
# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    # print(csv_file)
    df = pd.read_csv(csv_file,header=None)

    # ignore dataframe if fms column contains only 0
    if( (df[1] == 0).all()):
        print(csv_file)
        print("dataframe diregarded")
        continue


    if len(df) < 400:
        print(f"Skipping {csv_file} — only {len(df)} rows")
        continue  # skip files that are too short
    
    df = df.iloc[:400]

    # Fill NaN values (forward fill, then backward fill if needed)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # If NaNs still remain (e.g., entire column is NaN), fill with 0
    df = df.fillna(0)

    # Extract features: all columns except index 1 (column 2)
    x_cols = [0] + list(range(2, 8))  # [0, 2, 3, 4, 5, 6]
    x = df.iloc[:, x_cols].values     # shape: (400, 6)
    
    # Extract target: column index 1
    y = df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed

    # Append
    X_list.append(x)
    y_list.append(y)  # or y[-1] if you want to predict the last value only

# Convert lists to arrays
X = np.stack(X_list)  # shape: (num_samples, 400, 6)
y = np.stack(y_list)  # shape: (num_samples, 400) or (num_samples,) depending on choice
y = y[:, :, np.newaxis]  # shape becomes (num_samples, 400, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Assume last column is the label
# X = X.iloc[:, :-1].values  # Converts to NumPy array
# y = y.iloc[:, -1].values   # Converts to NumPy array (integer labels)



feature_idx = 3  # choose the feature you want to plot (0 to 6)
sample_indices = np.random.choice(429, size=10, replace=False)  # pick 10 random samples

plt.figure(figsize=(12, 5))

for idx in sample_indices:
    plt.plot(X[idx, :, feature_idx], label=f'Sample {idx}')

plt.title(f"Feature {feature_idx} over Time for 10 Samples")
plt.xlabel("Timestep")
plt.ylabel("Feature Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
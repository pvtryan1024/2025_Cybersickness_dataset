import pandas as pd
import glob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
# Repeat static input across time (broadcast it to match 420 steps)
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
    Concatenate,
    RepeatVector
)
from sklearn.model_selection import train_test_split
import dtw
from dtw import *

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


    if len(df) < 420:
        print(f"Skipping {csv_file} — only {len(df)} rows")
        continue  # skip files that are too short
    
    df = df.iloc[:420]

    # Fill NaN values (forward fill, then backward fill if needed)
    # df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.ffill().bfill()

    # If NaNs still remain (e.g., entire column is NaN), fill with 0
    df = df.fillna(0)

    if len(df.columns) > 12 or len(df.columns) < 11:
        print(f"Skipping {csv_file} — too many {len(df.columns)} features /////////////////////////////")


    # Extract features: all columns except index 1 (column 2)
    # print(csv_file)
    # x_cols = [0] + list(range(2, 11))  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = df.iloc[:, x_cols].values     # shape: (400, 6)
    print(x.shape)
    # print(x[10][10])
    column_values = x[:, 10]

    # Find indices where the value is greater than 100
    indices = np.where(column_values > 100)[0]

    # Print matching rows
    if len(indices) >0 :
        print("something wrong with age value" + csv_file)

    # Extract target: column index 1
    y = df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed
    X_list.append(x)
    y_list.append(y)  # or y[-1] if you want to predict the last value only

# Convert lists to arrays
X = np.stack(X_list)  # shape: (num_samples, 400, 6)
y = np.stack(y_list)  # shape: (num_samples, 400) or (num_samples,) depending on choice
y = y[:, :, np.newaxis]  # shape becomes (num_samples, 400, 1)

print(X.dtype)  # or whatever your input array is called
print(X.shape)  # or whatever your input array is called
print(type(X[0][0]))  # check inner element
print(X[0][0])
# print("X shape:", X.shape)
# print("y shape:", y.shape)


# Example: x shape = (429, 420, 10)
# Replace this with your actual dataset variable

# Flatten over samples and time: shape becomes (429*420, 10)
x_flat = X.reshape(-1, X.shape[2])


feature_names = [
    "FMS", "Accel X", "Accel Y", "Accel Z",
    "Angular Vel X", "Angular Vel Y", "Angular Vel Z", "Gender", "MSSQ" , "Age"
]
# Convert to DataFrame for seaborn
df = pd.DataFrame(x_flat, columns=[
    "Time", "FMS", "Accel X", "Accel Y", "Accel Z",
    "Angular Vel X", "Angular Vel Y", "Angular Vel Z", "Gender", "MSSQ" , "Age"
])

# Optional: if you have gender as a 0/1 column and want to label it:
# df['Gender'] = np.random.randint(0, 2, size=len(df))  # if needed
# Plot each feature as a separate violin plot
num_features = df.shape[1]
cols = 4  # Number of columns in the subplot grid
rows = int(np.ceil(num_features / cols))

fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

for i, feature in enumerate(feature_names):
    row = i // cols
    col = i % cols
    ax = axs[row, col] if rows > 1 else axs[col]
    
    sns.violinplot(y=df[feature], ax=ax, inner='box', color='skyblue')
    ax.set_title(feature)
    ax.set_xlabel("")
    ax.set_ylabel("Value")

# Hide any unused subplots
for j in range(i + 1, rows * cols):
    row = j // cols
    col = j % cols
    ax = axs[row, col] if rows > 1 else axs[col]
    ax.axis('off')

plt.tight_layout()
plt.suptitle("Violin Plots of Individual Features", fontsize=16, y=1.02)
plt.show()

age_values = X[:, :, 10]  # shape: (427, 420)

# Flatten to 1D array
age_flat = age_values.flatten()

# Get min and max
age_min = np.min(age_flat)
age_max = np.max(age_flat)

print(f"Age - Min: {age_min}, Max: {age_max}")
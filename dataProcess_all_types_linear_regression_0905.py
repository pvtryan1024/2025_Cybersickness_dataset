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
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import plot_model
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
    RepeatVector,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D
)
from sklearn.model_selection import train_test_split
import dtw
from dtw import *

dir_path = os.path.dirname(os.path.realpath(__file__))
# Get a list of all CSV files in a directory
csv_files = glob.glob(dir_path + '/*.csv')

# default_csv = glob.glob(dir_path +"/default" + '/*.csv')

# print(default_csv)

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# axis_df = pd.read_csv(default_csv[0], header=None)
# axis_df = axis_df[[8,9,10,11]]
# print(axis_df)
def calculate_smape(actual, predicted):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual (list or np.array): A list or NumPy array of actual values.
        predicted (list or np.array): A list or NumPy array of predicted values.

    Returns:
        float: The SMAPE value as a percentage.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate the absolute differences between predicted and actual values
    abs_diff = np.abs(predicted - actual)

    # Calculate the sum of absolute actual and predicted values, divided by 2
    denominator = (np.abs(actual) + np.abs(predicted)) / 2

    # Handle cases where the denominator might be zero to avoid division errors
    # A common approach is to set the error to 0 for such points if both actual and predicted are 0,
    # or to a very large number if one is non-zero and the other is zero.
    # For simplicity here, we'll avoid division by zero by setting the contribution to 0 if denominator is 0.
    # More robust handling might involve a small epsilon.
    smape_components = np.zeros_like(abs_diff, dtype=float)
    non_zero_denominator_indices = denominator != 0
    smape_components[non_zero_denominator_indices] = abs_diff[non_zero_denominator_indices] / denominator[non_zero_denominator_indices]

    # Calculate the mean of the components and multiply by 100 for percentage
    smape = np.mean(smape_components) * 100

    return smape


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

vectorized_bin_value = np.vectorize(bin_value)


def male_female_To_Value(value):
    if(value == 'f'):
        return 0
    elif(value == 'm'):
        return 1
    else:
        print(value)
        return 2

numerized_gender_value = np.vectorize(male_female_To_Value)

# samples = []
# X_list = []
# y_list = []
# # Loop through each CSV file and append its contents to the combined dataframe
# for csv_file in csv_files:
#     # print(csv_file)
#     df = pd.read_csv(csv_file,header=None)

#     # ignore dataframe if fms column contains only 0
#     if( (df[1] == 0).all()):
#         print(csv_file)
#         print("dataframe diregarded")
#         continue


#     if len(df) < 420:
#         print(f"Skipping {csv_file} — only {len(df)} rows")
#         continue  # skip files that are too short

#     df = df.iloc[:420]

#     # Fill NaN values (forward fill, then backward fill if needed)
#     # df = df.fillna(method='ffill').fillna(method='bfill')
#     df = df.ffill().bfill()

#     # If NaNs still remain (e.g., entire column is NaN), fill with 0
#     df = df.fillna(0)

#     if len(df.columns) > 12 or len(df.columns) < 11:
#         print(f"Skipping {csv_file} — too many {len(df.columns)} features /////////////////////////////")


#     # Extract features: all columns except index 1 (column 2)
#     print(csv_file)
#     x_cols = [0] + list(range(2, 11))  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     x = df.iloc[:, x_cols].values     # shape: (400, 6)

#     cat_column = df.iloc[:, 8].values  # assuming column 8 is 'f'/'m'
#     print(cat_column[0])

#     cat_onehot = np.array([[1, 0] if val == 'f' else [0, 1] for val in cat_column])

#     # Step 4: Drop the original categorical column from x
#     x_numeric = np.delete(x, 7, axis=1)  # shape: (420, 9)

#     # Step 5: Concatenate numeric and one-hot columns
#     x_final = np.concatenate([x_numeric, cat_onehot], axis=1)  # shape: (420, 9 + 2) = (420, 11)
#     x_final = x_final.astype(np.float32)

#     # x_final is your final feature array
#     # print("Final x shape:", x_final.shape)

#     # Extract target: column index 1
#     y = df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed
#     # y = vectorized_bin_value(y)


#     string_mask = np.vectorize(lambda d: isinstance(d, str))(x_final)

#     # Print positions
#     indices = np.argwhere(string_mask)
#     for i, j in indices:
#         print(f"String found at ({i}, {j}): {x_final[i, j]}")

#     print(x_final[0])
#     # Append
#     X_list.append(x_final)
#     y_list.append(y)  # or y[-1] if you want to predict the last value only


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

    # Extract features: all columns except index 1 (column 2)
    print(csv_file)
    x_cols = [0] + list(range(1, 11))  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = df.iloc[:, x_cols].values     # shape: (400, 6)

    cat_column = df.iloc[:, 8].values  # assuming column 8 is 'f'/'m'
    print(cat_column[0])

    cat_onehot = np.array([[1, 0] if val == 'f' else [0, 1] for val in cat_column])

    # Step 4: Drop the original categorical column from x
    x_numeric = np.delete(x, 8, axis=1)  # shape: (420, 9)

    # Step 5: Concatenate numeric and one-hot columns
    x_final = np.concatenate([x_numeric, cat_onehot], axis=1)  # shape: (420, 9 + 2) = (420, 11)
    x_final = x_final.astype(np.float32)

    # x_final is your final feature array
    # print("Final x shape:", x_final.shape)

    df = pd.DataFrame(x_final)
    # print(df)
    # print("//////////")
    combined_df = pd.concat([combined_df, df],axis=0,ignore_index=True)
    combined_df[0] = combined_df[0].fillna(0)
    combined_df[1] = combined_df[1].fillna(0)
    combined_df[2] = combined_df[2].fillna(0)
    combined_df[3] = combined_df[3].fillna(0)
    combined_df[4] = combined_df[4].fillna(0)
    combined_df[5] = combined_df[5].fillna(0)
    combined_df[6] = combined_df[6].fillna(0)
    combined_df[7] = combined_df[7].fillna(0)
    combined_df[8] = combined_df[8].fillna(0)
    combined_df[9] = combined_df[9].fillna(0)
    combined_df[10] = combined_df[10].fillna(0)
    combined_df[11] = combined_df[11].fillna(0)
    # combined_df[8] = numerized_gender_value(combined_df[8])


print(combined_df)




X= combined_df[[0,2,3,4,5,6,7,8,9,10,11]]
y= combined_df[1]


# print(X.dtype)  # or whatever your input array is called
print(type(X[0][0]))  # check inner element
print(X[0][0])
# print("X shape:", X.shape)
# print("y shape:", y.shape)



X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


# Make predictions using the testing set
sickness_y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: ", regr.coef_)
# # The mean absolute error
# print("Mean Absolute error:%.2f" % mean_absolute_error(y_test, sickness_y_pred))
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, sickness_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, sickness_y_pred))

# print(" ",  regr.score(X_test,y_test))
rmse = np.sqrt(mean_squared_error(y_test, sickness_y_pred))
mae = mean_absolute_error(y_test, sickness_y_pred)
r2 = r2_score(y_test, sickness_y_pred)
smape_value = calculate_smape(y_test, sickness_y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"R² Score:  {r2:.4f}")
print(f"Test SMAPE: {smape_value:.2f}%")


sampleIndex = [10,20,30,40,50,60,70,80,90,100]

def calculate_dtw(series1, series2):
    # DTW 유사도 계산
    dtw_result = dtw(series1, series2, keep_internals=True)
    return dtw_result


dtw_result_list = []
distanceSum = 0





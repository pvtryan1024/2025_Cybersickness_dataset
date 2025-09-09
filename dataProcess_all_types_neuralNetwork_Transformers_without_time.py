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
    print(csv_file)
    x_cols = [0] + list(range(2, 11))  # [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = df.iloc[:, x_cols].values     # shape: (400, 6)
    
    cat_column = df.iloc[:, 8].values  # assuming column 8 is 'f'/'m'
    print(cat_column[0])

    cat_onehot = np.array([[1, 0] if val == 'f' else [0, 1] for val in cat_column])

    # Step 4: Drop the original categorical column from x
    x_numeric = np.delete(x, 7, axis=1)  # shape: (420, 9)  

    # Step 5: Concatenate numeric and one-hot columns
    x_final = np.concatenate([x_numeric, cat_onehot], axis=1)  # shape: (420, 9 + 2) = (420, 11)
    x_final = x_final.astype(np.float32)

    # x_final is your final feature array
    # print("Final x shape:", x_final.shape)

    # Extract target: column index 1
    y = df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed
    # y = vectorized_bin_value(y)


    string_mask = np.vectorize(lambda d: isinstance(d, str))(x_final)

    # Print positions
    indices = np.argwhere(string_mask)
    for i, j in indices:
        print(f"String found at ({i}, {j}): {x_final[i, j]}")
        
    print(x_final[0])
    # Append
    X_list.append(x_final)
    y_list.append(y)  # or y[-1] if you want to predict the last value only

# Convert lists to arrays
X = np.stack(X_list)  # shape: (num_samples, 400, 6)
y = np.stack(y_list)  # shape: (num_samples, 400) or (num_samples,) depending on choice
y = y[:, :, np.newaxis]  # shape becomes (num_samples, 400, 1)

print(X.dtype)  # or whatever your input array is called
print(type(X[0][0]))  # check inner element
print(X[0][0])
# print("X shape:", X.shape)
# print("y shape:", y.shape)


# --- Hyperparameters ---
time_steps = 420
time_features = 6
static_features = 4
model_dim = 64
num_heads = 16
dropout_rate = 0.2


# --- Inputs ---
time_input = Input(shape=(time_steps, time_features), name='time_series_input')   # (420, 6)
static_input = Input(shape=(static_features,), name='static_input')               # (4,)


# --- Transformer Block ---
x = Dense(model_dim)(time_input)  # Project to model_dim
attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)(x, x)
x = LayerNormalization()(x + attn_output)  # Residual connection + norm

ffn_output = Dense(model_dim, activation='relu')(x)
ffn_output = Dropout(dropout_rate)(ffn_output)
x = LayerNormalization()(x + ffn_output)

# --- Global Pooling (combine time steps) ---
x = GlobalAveragePooling1D()(x)  # Shape: (batch, model_dim)

# --- Process static input ---
s = Dense(16, activation='relu')(static_input)

# --- Concatenate time + static features ---
combined = Concatenate()([x, s])

# --- Output layer for regression ---
# output = Dense(1, name='regression_output')(combined)
hidden0 = Dense(64, activation='relu')(combined)
hidden1 = Dense(128, activation='relu')(hidden0)
hidden2 = Dense(64, activation='relu')(hidden1)

# --- Output layer for regression ---
# output = Dense(1, name='regression_output')(combined)
output = Dense(420, name='regression_output')(hidden2)

# --- Build model ---
model = Model(inputs=[time_input, static_input], outputs=output)

# --- Compile ---
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Summary ---
model.summary()

# Time-series features (1-6)
X_ts = X[:, :, 1:7]  # shape: (429, 420, 6)

# Static features (last 3 columns, just from first timestep since they don't vary)
X_static = X[:, 0, 7:]  # shape: (429, 3)

# Split into numeric and categorical parts
X_numeric = X_static[:, :2]  # age and survey
X_other   = X_static[:, 2:]  # gender or any other static features

# Normalize numeric columns
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)


# Define columns to clip
clip_cols = [1, 2, 3]

# Define clipping thresholds (per feature)
clip_min = -25
clip_max = 25

# Clip only the selected features
X_clipped = X_ts.copy()
X_clipped[:, :, clip_cols] = np.clip(X_ts[:, :, clip_cols], clip_min, clip_max)


# xtsShape = X_ts.shape
# X_ts_reformed = X_ts.reshape(-1, 7)  # shape: (429*420, 7)

# # Apply scaler per feature
# scaler = MinMaxScaler()
# X_ts_reformed = scaler.fit_transform(X_ts_reformed)  # still shape: (429*420, 7)

# # Reshape back to original shape
# X_ts = X_ts_reformed.reshape(xtsShape)

print(X_numeric[0])
print(X_numeric_scaled[0])
print(X_other[0])


# Combine back together
X_static_scaled = np.concatenate([X_numeric_scaled, X_other], axis=1)


X_ts_train, X_ts_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_clipped, X_static_scaled, y, test_size=0.2, random_state=42
)

# # model_cnn.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
# model.fit(
#     x={'time_series_input': X_ts_train, 'static_input': X_static_train},
#     y=y_train,
#     validation_data=(
#         {'time_series_input': X_ts_test, 'static_input': X_static_test},
#         y_test
#     ),
#     batch_size=32,
#     epochs=20
# )

# y = your target variable, shape: (429,)
history = model.fit([X_clipped, X_static_scaled], y, epochs=30, batch_size=16, validation_split=0.2)

# y_pred = model_cnn.predict(X_test)
# y_pred = model.predict(X_test)
y_pred = model.predict({
    'time_series_input': X_ts_test,
    'static_input': X_static_test
})
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

print(y_pred.shape)
print(y_test.shape)

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)
smape_value = calculate_smape(y_test_flat, y_pred_flat)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"R² Score:  {r2:.4f}")
print(f"Test SMAPE: {smape_value:.2f}%")


y_pred1 = model.predict({
    'time_series_input': X_ts,
    'static_input': X_static_scaled
})
y_pred_flat1 = y_pred1.flatten()
y_flat = y.flatten()

sampleIndex = [10,20,30,40,50,60,70,80,90,100]

def calculate_dtw(series1, series2):
    # DTW 유사도 계산
    dtw_result = dtw(series1, series2, keep_internals=True)
    return dtw_result


dtw_result_list = []
distanceSum = 0

for i in sampleIndex:
    dtw_result = calculate_dtw(y[i].squeeze(), y_pred1[i].squeeze())
    dtw_result_list.append(dtw_result)
    print("////////////////////"+ str(i)+"//////////////////////////////")
    # print(dtw_result.path)
    print(dtw_result.distance)
    distanceSum += dtw_result.distance
    plt.plot(y[i].squeeze(), label='True')
    plt.plot(y_pred1[i].squeeze(), label='Predicted')
    plt.ylim(0, 20)
    plt.title("Transformer Comparison (Sample " + str(i) + ")")
    plt.legend()
    plt.show()

print("total distance sum is : " + str(distanceSum) )

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


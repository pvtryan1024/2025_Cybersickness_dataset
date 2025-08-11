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


# Inputs
ts_input = Input(shape=(420, 7), name='time_series_input')    # dynamic input
static_input = Input(shape=(4,), name='static_input')         # static input

# LSTM layers
x = LSTM(128, return_sequences=True)(ts_input)  # keep sequence output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

# x = LSTM(128, return_sequences=True)(x)
# x = Dropout(0.2)(x)
# x = BatchNormalization()(x)

x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

x = LSTM(32, return_sequences=True)(x)
x = Dropout(0.2)(x)

# Repeat static input across 420 timesteps
repeated_static = RepeatVector(420)(static_input)  # shape: (batch, 420, 3)

# Merge time-dependent LSTM output with repeated static input
combined = Concatenate(axis=-1)([x, repeated_static])  # shape: (batch, 420, 64+3)

# combined = Dense(64, activation='relu')(combined)
# combined = BatchNormalization()(combined)
# combined = Dropout(0.2)(combined)

# Output layer: predict one value per timestep
output = TimeDistributed(Dense(1))(combined)  # shape: (batch, 420, 1)

# Build and compile the model
model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Time-series features (first 8 columns)
X_ts = X[:, :, :7]  # shape: (429, 420, 8)

# Static features (last 3 columns, just from first timestep since they don't vary)
X_static = X[:, 0, 7:]  # shape: (429, 3)

# Split into numeric and categorical parts
X_numeric = X_static[:, :2]  # age and survey
X_other   = X_static[:, 2:]  # gender or any other static features

# Normalize numeric columns
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

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


num_samples, num_timesteps, num_features = X_ts.shape
x_reshaped = X_ts.reshape(-1, num_features)  # shape: (429*420, 7)

# Split into groups
x_time = x_reshaped[:, [0]]
x_accel = x_reshaped[:, 1:4]
x_gyro = x_reshaped[:, 4:7]

# Initialize scalers for each group
scaler_time = MinMaxScaler()
scaler_accel = MinMaxScaler()
scaler_gyro = MinMaxScaler()

# Fit-transform each group
x_time_scaled = scaler_time.fit_transform(x_time)
x_accel_scaled = scaler_accel.fit_transform(x_accel)
x_gyro_scaled = scaler_gyro.fit_transform(x_gyro)

# Concatenate back
x_scaled  = np.concatenate([x_time_scaled, x_accel_scaled, x_gyro_scaled], axis=1)  # shape: (429*420, 7)
X_ts = x_scaled.reshape(num_samples, num_timesteps, num_features)

X_ts_train, X_ts_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_ts, X_static_scaled, y, test_size=0.2, random_state=42
)

# model_cnn.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.fit(
    x={'time_series_input': X_ts_train, 'static_input': X_static_train},
    y=y_train,
    validation_data=(
        {'time_series_input': X_ts_test, 'static_input': X_static_test},
        y_test
    ),
    batch_size=32,
    epochs=20
)

# y_pred = model_cnn.predict(X_test)
# y_pred = model.predict(X_test)
y_pred = model.predict({
    'time_series_input': X_ts_test,
    'static_input': X_static_test
})
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"R² Score:  {r2:.4f}")

for i in range(20):
    plt.plot(y_test[i].squeeze(), label='True')
    plt.plot(y_pred[i].squeeze(), label='Predicted')
    plt.ylim(0, 20)
    plt.title("LSTM Comparison (Sample " + str(i) + ")")
    plt.legend()
    plt.show()

# plt.plot(y_test[0].squeeze(), label='True')
# plt.plot(y_pred[0].squeeze(), label='Predicted')
# plt.title("1D CNN Prediction (Sample 0)")
# plt.legend()
# plt.show()
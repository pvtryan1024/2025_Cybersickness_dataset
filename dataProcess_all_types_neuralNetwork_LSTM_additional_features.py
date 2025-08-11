import pandas as pd
import glob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


model_cnn = Sequential([
    LSTM(64, return_sequences=True, input_shape=(420, 11)),  # must return sequence
    Dense(32, activation='relu'),
    TimeDistributed(Dense(1))  # Output: (batch_size, 400, 1)
])

model_cnn.compile(optimizer='adam', loss='mse')
model_cnn.summary()


# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]



batch_size = 32
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(batch_size).repeat()

model_cnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


y_pred = model_cnn.predict(X_test)
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"R² Score:  {r2:.4f}")

for i in range(10):
    plt.plot(y_test[i].squeeze(), label='True')
    plt.plot(y_pred[i].squeeze(), label='Predicted')
    plt.title("LSTM Prediction (Sample " + str(i) + ")")
    plt.legend()
    plt.show()

# plt.plot(y_test[0].squeeze(), label='True')
# plt.plot(y_pred[0].squeeze(), label='Predicted')
# plt.title("1D CNN Prediction (Sample 0)")
# plt.legend()
# plt.show()
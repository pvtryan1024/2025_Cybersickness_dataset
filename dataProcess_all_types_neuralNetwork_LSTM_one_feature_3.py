import pandas as pd
import glob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

# default_csv = glob.glob(dir_path +"/default" + '/*.csv')

# print(default_csv)

# # Create an empty dataframe to store the combined data
# combined_df = pd.DataFrame()

# axis_df = pd.read_csv(default_csv[0], header=None)
# axis_df = axis_df[[8,9,10,11]]
# print(axis_df)

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
    df = df.ffill().bfill()

    # Extract features: all columns except index 1 (column 2)
    # x_cols = [0] + list(range(2, 8))  # [0, 2, 3, 4, 5, 6, 7]
    # x_cols = list(range(2, 8))  # [0, 2, 3, 4, 5, 6, 7 ]
    x_cols = [3]  # [0, 2, 3, 4, 5, 6, 7]
    x = df.iloc[:, x_cols].values     # shape: (400, 6)
    
    # Extract target: column index 1
    y = df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed

    # Append
    X_list.append(x)
    y_list.append(y)  # or y[-1] if you want to predict the last value only


    # df = pd.concat([df.reset_index(drop=True),axis_df.reset_index(drop=True)],axis=1)
    # df[8] = df[8].fillna(0)
    # df[9] = df[9].fillna(0)
    # df[10] = df[10].fillna(0)
    # df[11] = df[11].fillna(0)

    # # fms 스코어 0-20 사이를 0-4 사이로 압축 binning
    # df['binned'] = df[1].apply(bin_value)
    
    # df = df.head(400)
    # # print(df)
    # # print("//////////")
    # # combined_df = pd.concat([combined_df, df],axis=0,ignore_index=True)
    # # combined_df[0] = combined_df[0].fillna(0)
    # # combined_df[1] = combined_df[1].fillna(0)
    # # combined_df[2] = combined_df[2].fillna(0)
    # # combined_df[3] = combined_df[3].fillna(0)
    # # combined_df[4] = combined_df[4].fillna(0)


    # # combined_df[5] = combined_df[5].fillna(0)
    # # combined_df[6] = combined_df[6].fillna(0)
    # # combined_df[7] = combined_df[7].fillna(0)

    # # combined_df[8] = combined_df[8].fillna(0)
    # # combined_df[9] = combined_df[9].fillna(0)
    # # combined_df[10] = combined_df[10].fillna(0)
    # # combined_df[11] = combined_df[11].fillna(0)

    # # combined_df['binned'] = combined_df['binned'].fillna(0)

    # df[0] = df[0].fillna(0)
    # df[1] = df[1].fillna(0)
    # df[2] = df[2].fillna(0)
    # df[3] = df[3].fillna(0)
    # df[4] = df[4].fillna(0)
    # df[5] = df[5].fillna(0)
    # df[6] = df[6].fillna(0)
    # df[7] = df[7].fillna(0)
    # df[8] = df[8].fillna(0)
    # df[9] = df[9].fillna(0)
    # df[10] = df[10].fillna(0)
    # df[11] = df[11].fillna(0)




# Print the combined dataframe
# print(combined_df)
# combined_df[['a','b','c','d']] = combined_df[6].str.split('.',n=4,expand=True)
# combined_df[6] = (combined_df['a']+'.'+combined_df['b']).astype(float)
# combined_df[7] = (combined_df['c']+'.'+combined_df['d']).astype(float)
# combined_df[7] = combined_df[7].fillna(0)

# print(combined_df)
# print(combined_df.dtypes)
# print(combined_df)

# 0 time , 1 linear x, 2 linear y, 3 linear z, 4 angular x, 5 angular y, 6 angular z
# X= combined_df[[0]]
# X= combined_df[[2]]
# X= combined_df[[3]]
# X= combined_df[[4]]
# X= combined_df[[5]]
# X= combined_df[[6]]

#roll 영향이 아주 약간 
# X= combined_df[[7]]

#
# X= combined_df[[8]]
# X= combined_df[[9]]
# X= combined_df[[10]]
# X= combined_df[[11]]
# X= combined_df[[8,9]]
# X= combined_df[[7]]
# X= combined_df[[0,7,8,9]]
# X= combined_df[[0,7,10,11]]
# X= combined_df[[0,2,3,4,5,6]]
# X = datastack[0,2,3,4,5,6]

# X= combined_df[[8]]

# y= combined_df['binned']
# y= combined_df[1]
# y = datastack[1]

# Convert lists to arrays
X = np.stack(X_list)  # shape: (num_samples, 400, 6)
y = np.stack(y_list)  # shape: (num_samples, 400) or (num_samples,) depending on choice
y = y[:, :, np.newaxis]  # shape becomes (num_samples, 400, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Assume last column is the label
# X = X.iloc[:, :-1].values  # Converts to NumPy array
# y = y.iloc[:, -1].values   # Converts to NumPy array (integer labels)



model_lstm = Sequential([
    # LSTM layers
    LSTM(128, return_sequences=True, input_shape=(400, 1)),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
      # must return sequence
    TimeDistributed(Dense(1))  # predict 1 value per timestep
])

model_lstm.compile(optimizer='adam', loss='mse')

# Model summary
model_lstm.summary()



# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]



batch_size = 32
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(batch_size).repeat()

model_lstm.fit(train_dataset, epochs=20, steps_per_epoch=steps_per_epoch)


#####################################################################
# y_pred = model_lstm.predict(X_test)  # shape: (samples, 400, 1)

# y_pred_flat = y_pred.flatten()
# y_test_flat = y_test.flatten()

# rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
# mae = mean_absolute_error(y_test_flat, y_pred_flat)
# r2 = r2_score(y_test_flat, y_pred_flat)

# print(f"Test RMSE: {rmse:.4f}")
# print(f"Test MAE:  {mae:.4f}")
# print(f"R² Score:  {r2:.4f}")


# per_timestep_rmse = np.sqrt(np.mean((y_pred - y_test)**2, axis=0))  # shape: (400, 1)
# plt.plot(per_timestep_rmse)
# plt.title("RMSE per timestep")
# plt.xlabel("Timestep")
# plt.ylabel("RMSE")
# plt.show()

# rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
# mae = mean_absolute_error(y_test_flat, y_pred_flat)
# r2 = r2_score(y_test_flat, y_pred_flat)

# print(f"Test RMSE: {rmse:.4f}")
# print(f"Test MAE:  {mae:.4f}")
# print(f"R² Score:  {r2:.4f}")

# plt.plot(y_test[0].squeeze(), label='True')
# plt.plot(y_pred[0].squeeze(), label='Predicted')
# plt.title("LSTM Prediction (Sample 0)")
# plt.legend()
# plt.show()
#####################################################################


y_pred = model_lstm.predict(X_test)  # shape: (samples, 400, 1)

y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()
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
    plt.ylim(0, 20)
    plt.title("LSTM Angular Velocity (Sample " + str(i) + ")")
    plt.legend()
    plt.show()
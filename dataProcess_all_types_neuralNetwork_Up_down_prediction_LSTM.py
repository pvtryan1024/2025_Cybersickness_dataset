import pandas as pd
import glob
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
import tensorflow as tf
import tensorflow.keras as keras
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
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score ,  ConfusionMatrixDisplay
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K



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
    x_cols = [0] + list(range(2, 8))  # [0, 2, 3, 4, 5, 6, 7]
    x = df.iloc[:, x_cols].values     # shape: (400, 7)
    

    # # fms 스코어 0-20 사이를 0-4 사이로 압축 binning
    # df[1] = df[1].apply(bin_value)



    # Extract target: column index 1
    y = df.iloc[:, 1].values        
    # shape: (400,) — or pick one value if needed
    # print(encoded_df)
    # y = encoded_df.iloc[:, 1].values          # shape: (400,) — or pick one value if needed


    # print("y")
    # print(y)
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

print(y)
print(y.shape)

y_class = (np.diff(y, axis=1) > 0).astype(int)  # shape: (429, 399)

# print(y_class)
print(y_class[0])
# print(y_class.shape)
# Optional: pad the first timestep to make it (429, 400) again
y_class = np.pad(y_class, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)


unique, counts = np.unique(y_class.flatten(), return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))



model_cnn = Sequential([
    LSTM(64, return_sequences=True, input_shape=(400, 7)),
    Dropout(0.3),  # reduce overconfident noisy signals
    # TimeDistributed(Dense(1))  # Output: (batch_size, 400, 1)
    TimeDistributed(Dense(1, activation='sigmoid'))
])




def weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        return K.mean(bce * weight_vector)
    return loss

# If imbalance is like 80% zeros, 20% ones
loss_fn = weighted_binary_crossentropy(zero_weight=1.0, one_weight=10.0 )


loss_label_smoothing = BinaryCrossentropy(label_smoothing = 0.1)

# model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model_cnn.compile(optimizer='adam', loss = loss_label_smoothing, metrics=['accuracy'])
model_cnn.compile(optimizer='adam', loss = loss_fn, metrics=['accuracy'])
model_cnn.summary()

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_class[:split], y_class[split:]



batch_size = 32
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(batch_size).repeat()

model_cnn.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = model_cnn.evaluate(X_test, y_test)
print(f"Binary classification accuracy: {accuracy:.2%}")


y_pred = model_cnn.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)


# Flatten both predictions and labels
y_pred_flat = y_pred_binary.flatten()
y_true_flat = y_test.flatten()


# Confusion matrix: [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()

# Metrics
sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
precision   = precision_score(y_true_flat, y_pred_flat)
recall      = recall_score(y_true_flat, y_pred_flat)
accuracy    = accuracy_score(y_true_flat, y_pred_flat)
f1          = f1_score(y_true_flat, y_pred_flat)

# Print results
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Accuracy:             {accuracy:.4f}")
print(f"F1 Score:             {f1:.4f}")


# Compare predictions vs ground truth for one sample
sample = 0
plt.plot(y_test[sample].squeeze(), label='True')
plt.plot(y_pred_binary[sample].squeeze(), label='Predicted')
plt.title("Up/Down Prediction - Sample 0")
plt.legend()
plt.show()

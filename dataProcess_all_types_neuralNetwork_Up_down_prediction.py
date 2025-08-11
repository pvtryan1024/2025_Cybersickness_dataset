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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score ,  ConfusionMatrixDisplay , classification_report
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight


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

# print(y)
# print(y.shape)
# y_raw shape = (num_samples, 400)
# We'll compare each value to the previous one
diff = np.diff(y, axis=1)

# Create class labels based on difference
y_class = np.where(diff > 0, 2,      # up
           np.where(diff < 0, 0, 1)) # down or same
# y_class = (np.diff(y, axis=1) > 0).astype(int)  # shape: (429, 399)

# print(y_class)
print(y_class[0])
# print(y_class.shape)
# Optional: pad the first timestep to make it (429, 400) again
y_class = np.pad(y_class, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_class[:split], y_class[split:]

# Flatten labels to compute class weights
y_flat = y_train.flatten()

# Compute weights using sklearn
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_flat), y=y_flat)
print("Class weights:", class_weights)




model_cnn = Sequential([
    Conv1D(64, kernel_size=25, activation='relu', padding='same', input_shape=(400, 7)),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(64, kernel_size=25, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(32, kernel_size=10, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.3),
    # TimeDistributed(Dense(1))  # Output: (batch_size, 400, 1)
    # TimeDistributed(Dense(1, activation='sigmoid'))
    TimeDistributed(Dense(3, activation='softmax'))  # 3 output classes
])


def weighted_sparse_categorical_crossentropy(weights):
    weights = K.constant(weights)
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        y_true = K.squeeze(y_true, axis=-1)  # shape: (batch, seq)
        
        # Gather the weight for each true class
        sample_weights = K.gather(weights, y_true)

        # Compute loss
        scce = K.sparse_categorical_crossentropy(y_true, y_pred)
        return K.mean(scce * sample_weights)
    return loss

def weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        return K.mean(bce * weight_vector)
    return loss

# If imbalance is like 80% zeros, 20% ones
loss_fn = weighted_binary_crossentropy(zero_weight=1.0, one_weight=(163437 / 8163) )


loss_label_smoothing = BinaryCrossentropy(label_smoothing = 0.1)

# model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model_cnn.compile(optimizer='adam', loss = loss_label_smoothing, metrics=['accuracy'])
# model_cnn.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_cnn.compile(
    optimizer='adam',
    loss=weighted_sparse_categorical_crossentropy(class_weights),
    metrics=['accuracy']
)
model_cnn.summary()





model_cnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model_cnn.evaluate(X_test, y_test)
print(f"Binary classification accuracy: {accuracy:.2%}")



# Predict class probabilities
y_pred_probs = model_cnn.predict(X_test)  # shape: (num_samples, 400, 3)
y_pred = np.argmax(y_pred_probs, axis=-1)  # shape: (num_samples, 400)
y_true = y_test.squeeze()

# # Flatten both predictions and labels
# y_pred_flat = y_pred_binary.flatten()
# y_true_flat = y_test.flatten()

print(classification_report(y_true.flatten(), y_pred.flatten(), digits=4))


unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique, ['Down (0)', 'Same (1)', 'Up (2)'])
plt.title('Class Distribution')
plt.show()


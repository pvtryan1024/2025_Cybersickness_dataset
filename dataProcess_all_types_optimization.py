import pandas as pd
import glob
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint


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

# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    # print(csv_file)
    df = pd.read_csv(csv_file,header=None)

    # ignore dataframe if fms column contains only 0
    if( (df[1] == 0).all()):
        print(csv_file)
        print("dataframe diregarded")
        continue

    df = pd.concat([df.reset_index(drop=True),axis_df.reset_index(drop=True)],axis=1)
    df[8] = df[8].fillna(0)
    df[9] = df[9].fillna(0)
    df[10] = df[10].fillna(0)
    df[11] = df[11].fillna(0)

    # fms 스코어 0-20 사이를 0-4 사이로 압축 binning
    df['binned'] = df[1].apply(bin_value)
    
    df = df.head(400)
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

    combined_df['binned'] = combined_df['binned'].fillna(0)

    
# Print the combined dataframe
# print(combined_df)
# combined_df[['a','b','c','d']] = combined_df[6].str.split('.',n=4,expand=True)
# combined_df[6] = (combined_df['a']+'.'+combined_df['b']).astype(float)
# combined_df[7] = (combined_df['c']+'.'+combined_df['d']).astype(float)
# combined_df[7] = combined_df[7].fillna(0)
combined_df[5] = combined_df[5] * 57.29578
combined_df[6] = combined_df[6] * 57.29578
combined_df[7] = combined_df[7] * 57.29578
print(combined_df)
print(combined_df.dtypes)
print(combined_df)

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
X= combined_df[[0,2,3,4,5,6]]

# X= combined_df[[8]]

y= combined_df['binned']

def find_threshold(df, column, percentile):
    """
    Find the threshold in a DataFrame column such that 80% of the data is positive and 20% is negative.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name for which to find the threshold.
    
    Returns:
    float: The threshold value.
    pd.DataFrame: DataFrame with an additional column for classification.
    """
    # Sort the DataFrame based on the column
    sorted_df = df.sort_values(by=column).reset_index(drop=True)
    
    # Calculate the index for the 80th percentile
    threshold_index = int(percentile * len(sorted_df))
    
    # Determine the threshold
    threshold = sorted_df.iloc[threshold_index][column]
    
    # Classify the data
    df['classification'] = df[column].apply(lambda x: 'positive' if x >= threshold else 'negative')
    
    return threshold, df


x0 = [1,1,1,1,1,1,1]

def calculate_new_column_activation_value(b_value):
    b_value_new = 0
    if (b_value < 72):
        # print(b_value)
        b_value_new = 32 * ( 1 - np.exp( -0.05 * b_value) )
    else:
        yOffset = 32 * (1 - np.exp(-0.05 * 72)) - 25
        b_value_new = yOffset * np.exp(-0.2*(b_value - 72))+ 25

    return b_value_new/8

def squish_into_sqrt(value):
    if (value <= 0):
        return 0
    else :
        return math.sqrt(value)

def Objective_Function(coefficient_array):
    # 0 - 32 사이 값을 0-4 사이로 압축
    combined_df['obj'] = ( 
        coefficient_array[0]*combined_df[5]*combined_df[5] + coefficient_array[1]*combined_df[6]*combined_df[6] + coefficient_array[2]*combined_df[7]*combined_df[7] 
        + coefficient_array[3]*combined_df[5]*combined_df[6] + coefficient_array[4]*combined_df[6]*combined_df[7] + coefficient_array[5]*combined_df[5]*combined_df[7]
        + coefficient_array[6]*combined_df[5]*combined_df[6]*combined_df[7]
    )
    combined_df['obj'] = combined_df['obj'].apply(squish_into_sqrt)

    combined_df['obj_new'] = combined_df['obj'].apply(calculate_new_column_activation_value)
    # fms와의 차이 절대값 찾아서 반환
    combined_df['minimize'] = ( combined_df['binned'] - combined_df['obj_new'] ) 
    combined_df['minimize'] = combined_df['minimize'].apply(abs)

    sum = combined_df['minimize'].sum()
    print(sum)
    return sum


parameters = minimize(Objective_Function, x0, bounds=((0,30),(0,30),(0,30),(0,30),(0,30),(0,30),(0,30)))
# parameters = minimize(Objective_Function, x0, bounds=((-10,30),(-10,30),(-10,30),(-10,30),(-10,30),(-10,30),(-10,30)))
# parameters = minimize(Objective_Function, x0, bounds=((-10,30),(-10,30),(-10,30),(-10,30),(-10,30),(-10,30),(-10,30)))

print(parameters)

###########################################################################################
combined_df.columns = combined_df.columns.astype(str)
X= combined_df[['0','obj_new']]
# X= combined_df[['0','7','8','9','10','11']]
y= combined_df['binned']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


# Make predictions using the testing set
sickness_y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: ", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, sickness_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, sickness_y_pred))

print("Shape : ",combined_df.shape)
# print(" ",  regr.score(X_test,y_test)) 
#########################################################################################

threshold, df_sorted = find_threshold(combined_df, 'obj_new', 0.8)
print(threshold)
print(df_sorted)
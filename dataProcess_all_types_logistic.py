import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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

    
# Print the combined dataframe
# print(combined_df)
# combined_df[['a','b','c','d']] = combined_df[6].str.split('.',n=4,expand=True)
# combined_df[6] = (combined_df['a']+'.'+combined_df['b']).astype(float)
# combined_df[7] = (combined_df['c']+'.'+combined_df['d']).astype(float)
# combined_df[7] = combined_df[7].fillna(0)

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

y= combined_df[1]

# print(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

regr = LogisticRegression(max_iter=500)
regr.fit(X_train, y_train)


# Make predictions using the testing set
sickness_y_pred = regr.predict(X_test)

# Use score method to get accuracy of model
score = regr.score(X_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, sickness_y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()


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


# threshold, df_sorted = find_threshold(combined_df, 9, 0.8)
# print(threshold)
# print(df_sorted)
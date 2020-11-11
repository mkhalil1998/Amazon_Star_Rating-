import pandas as pd
import numpy as np 

# Preprocess Train.csv 
# Find new features for modeling

#####################################################

# Reading train.csv file 
train_fpath = r'/Users/mac/Desktop/train.csv'
train_DF = pd.read_csv(train_fpath)

# Dropping duplicate records in case if there were any
train_DF.drop_duplicates(inplace=True)
# print(train_DF)

# --------------------
# Function that determines number of exclamations points in both summary and text

def find_number_exclamation(summary,text):
    if type(text) != str or type(summary) != str:
        return 0
    else:
        return summary.count('!')+text.count('!')

# Adds this as a new feature to the train_DF 
train_DF['Number of Exclamations']= train_DF.apply(lambda x: find_number_exclamation(x.Summary,x.Text),axis=1)

#print(train_DF)

# --------------------
# Function that determines the number of words in a string
# It is used for both summary and text 

def number_of_words(text):
    if type(text) != str:
        return 0
    else:
        return len(text.split())
# Finds the number of words in summary and adds it as a new feature 
train_DF['# of words in Summary']=train_DF.apply(lambda x: number_of_words(x.Summary),axis=1)
# Finds the number of words in text and adds it as a new feature 
train_DF['# of words in Text']=train_DF.apply(lambda x: number_of_words(x.Text),axis=1)

# print(train_DF)

# --------------------

# Adding new feature: Average score for each specific movie 
# Computes the mean score for the ratings for a specific movie 
avg_score_movie = train_DF.groupby(['ProductId'])['Score'].mean()
# Adds new feature to the train_DF
train_DF['Avg Score']= train_DF.apply(lambda x: avg_score_movie.loc[x.ProductId],axis=1)
# Fill NaNs with mean score
train_DF['Avg Score']=train_DF['Avg Score'].fillna(train_DF['Avg Score'].mean())

#print(train_DF)
# -------------------

# Adding new feature:

def find_number_positive_words(summary,text):
    if type(text) != str or type(summary) != str:
        return 0
    else:
        return summary.count('good'or'original'or 'powerful' or 'enjoyable')+text.count('good'or 'original'or 'powerful' or 'enjoyable')

# Adds this as a new feature to the train_DF 
train_DF['Positive']= train_DF.apply(lambda x: find_number_positive_words(x.Summary,x.Text),axis=1)

# --------------------

def find_number_negative_words(summary,text):
    if type(text) != str or type(summary) != str:
        return 0
    else:
        return summary.count('boring'or 'confused' or 'static'or 'slow')+text.count('boring'or 'confused' or 'static'or 'slow')

# Adds this as a new feature to the train_DF 
train_DF['Negative']= train_DF.apply(lambda x: find_number_negative_words(x.Summary,x.Text),axis=1)

# --------------------
# Creating a new column called helpfulness which divides numerator by denominator 
train_DF['Helpfulness'] = train_DF['HelpfulnessNumerator'] / train_DF['HelpfulnessDenominator']
# Replaces all NaNs by zeros
train_DF['Helpfulness'] = train_DF['Helpfulness'].replace(np.nan, 0)
print(train_DF)

# --------------------

# Converting a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
# Creating a count vector object
count_vect = CountVectorizer()

# Fitting and transforming the 'Text' column in X_train to count vectors
train_text_counts = count_vect.fit_transform((train_DF['Summary']).values.astype('U'))

# Shape of the sparse matrix
print(train_text_counts.shape)

# --------------------

# Transforming a count matrix to a normalized tf-idf representation
from sklearn.feature_extraction.text import TfidfTransformer

# Creating a tf-idf transformer object
tf_transformer = TfidfTransformer().fit(train_text_counts)
# Transfroming X_train_text_counts to tf-idf
train_text_tf = tf_transformer.transform(train_text_counts)

# Printing the shapes of tf-idf data
print(train_text_tf.shape)

# --------------------
'''''
# It is commented out because it takes a lot of computational time 
# Function to compute the optimal number of components of TruncatedSVD

def optimal_svd_components():
  from sklearn.decomposition import TruncatedSVD

  # Initializing the no. of components
  n_comps = 1
  # Initializing the sum of Explained Variance Ratio
  sum_evr = 0
  # Initial step size when the sum_evr is less 0.985
  step = 49

  # TruncatedSVD to preserve 0.99% of the ratio of variance
  while sum_evr < 0.95:
    svd = TruncatedSVD(n_components=n_comps, n_iter=10, random_state=42)
    # X with reduced dimensions
    rX_train_text_tf = svd.fit_transform(X_train_text_tf)
    sum_evr = svd.explained_variance_ratio_.sum()
    print(f'No. of Components: {n_comps:<15} Sum of Explained Variance Ratio: {sum_evr}')
    
    # When sum_evr is greater than 0.9 then increment the n_comps by 1
    # When sum_evr is greater than 0.8 then increment the n_comps by step//2
    # When sum_evr is less than 0.8 then increment the n_comps by step
    if sum_evr > 0.9:
      n_comps += 1
    elif sum_evr > 0.8:
      n_comps += step//2
    else:
      n_comps += step
    
    # When sum_evr is greater than 0.99 then decrement the n_comps by 1
    if sum_evr > 0.95:
      n_comps -= 1
    
  # Printing the optimal no. of components to be used in TruncatedSVD
  print(f'Optimal No. of Components: {n_comps:<15} Sum of Explained Variance Ratio: {sum_evr}')
  return n_comps

train_DF.shape

'''
# ----------------------

# Dimensionality reduction using truncated SVD (aka LSA)
from sklearn.decomposition import TruncatedSVD

# Creating a TruncatedSVD object
svd = TruncatedSVD(n_components=30, n_iter=10, random_state=42)

# X_train with reduced dimensions
r_train_text_tf = svd.fit_transform(train_text_tf)

# Converting numpy array to dataframes
train_text_tf = pd.DataFrame(r_train_text_tf, index=train_DF.index)

# Printing the shapes of X_train_text_tf
print(train_text_tf.shape)


# Concatenating the dataframe created in the previous block to X_train
training_set = pd.concat([train_DF, train_text_tf], axis=1)

# Printing the shapes of X_train and X_test
print(training_set.shape)

# --------------------

# Identifying rows in train_DF that has 'Score' as null. These rows correspond to the rows in the test.csv.
test_rows = training_set['Score'].isnull()
# Filtering the test records from train_DF
test_DF = training_set[test_rows]

# Filtering the train records from train_DF
training_set_f = training_set[test_rows != True]
# Dropping rows that contain NAs
training_set_f= training_set_f.dropna()
# Checking if there are anymore null values
training_set_f.isnull().sum()

# --------------------

# Saving preprocessed data into csv file to be used for modeling 

training_set_f.to_csv("training_set_processed.csv")
test_DF.to_csv("prediction_set.csv")
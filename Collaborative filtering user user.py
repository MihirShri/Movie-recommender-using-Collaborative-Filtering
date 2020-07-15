"""
Author @ Mihir_Srivastava
Dated - 12-07-2020
File - Movie_recommender_collaborative_filtering
Aim - To recommend top 5 movies using user based collaborative filtering technique.
"""

# Import necessary libraries
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# Read csv file to get movie ID and movie names
df1 = pd.read_csv('u.item', sep='|', usecols=[0, 1], names=['MovieId', 'Movie Name'], engine='python')
df1.set_index('MovieId', inplace=True)

# Read csv file to get the userID, movieID and ratings
df = pd.read_csv('u.data', sep='\t', usecols=[0, 1, 2], names=['UserID', 'MovieID', 'Ratings'], engine='python')

# Create a pivot table with UserID as rows and MovieID as columns and each cell (x, y) represents the rating given by
# a user x to a movie y.
df_new = df.pivot(index='UserID', columns='MovieID', values='Ratings').fillna(0)  # dim(df_new) = (943, 1682)

# Convert the pivot table to an array
ratings = np.array(df_new)

# Get a sparse matrix which only consists of the non zero data (this is done to avoid unnecessary processing)
matrix = csr_matrix(df_new.values)

# Calculate the similarity between each user using the cosine similarity matrix
cosine_sim = cosine_similarity(matrix, matrix)  # dim(cosine_sim) = (943, 943)


# A function to split the data into train set and test set (to be used to calculate the accuracy of our model)
def train_test_split(rating):
    # Initially fill the test set matrix with all zeros
    test = np.zeros(ratings.shape)  # dim(test) = (943, 1682)

    train = ratings.copy()  # dim(train) = (943, 1682)

    # for each user
    for user in range(ratings.shape[0]):  # loop from 0 to (943 - 1)

        # Select the indices of 10 MovieIDs which have been rated by the user (nonzero rating)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=10, replace=False)

        # In the train set, fill those indices with 0 (since we have to predict these and then compare with the test set)
        train[user, test_ratings] = 0.

        # In the test set, fill those indices with the rating given by the user (this will be our actual value which will be compared to the predicted value)
        test[user, test_ratings] = ratings[user, test_ratings]

    # Ensure that test and train sets are truly disjoint
    assert (np.all((train * test) == 0))

    return train, test


# Split the data
train, test = train_test_split(ratings)


# A function to predict the rating a user would give to a movie (which he has not rated yet) based on the weighted
# average of all other users (weighted because users who are more similar to a user x will be given more weight than the
# users who are not so similar to user x)
def predict_fast_simple(rating, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(rating) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# Calculate cosine similarity for the train set
train_cosine_sim = cosine_similarity(train, train)


# A function to get the mean squared error between the predicted and actual value
def get_mse(pred, actual):
    # Only considering the nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()

    return mean_squared_error(pred, actual)


# Make prediction on the train set
user_prediction = predict_fast_simple(train, train_cosine_sim, 'user')

print()
print("The model has an accuracy of " + str(100 - get_mse(user_prediction, test)) + "%")

# Now make predictions using the actual DataSet
dff = pd.DataFrame(predict_fast_simple(ratings, cosine_sim, 'user'))

# Some preprocessing (not required)
dff.columns += 1
dff.index += 1
dff.index.rename('User ID', inplace=True)


# A function which recommends movies to a user, given his userID
def Recommend(UserID):
    result = {}
    for j in df_new.columns:
        if df_new.iloc[UserID - 1][j] == 0 and dff.iloc[UserID - 1][j] != 0:
            # The dictionary result contains keys as the movies which were not initially rated by the user (whose values
            # we predicted) and values as the corresponding predicted rating
            result.__setitem__(j, dff.iloc[UserID - 1][j])

    # Sort the dictionary wrt to values i.e. in decreasing order of the predicted ratings
    h = sorted(result.items(), key=lambda x: x[1], reverse=True)

    # Consider only the top 5 movies with the highest predicted ratings
    h = h[:5]

    # Take the MovieIDs of the top 5 movies
    result1 = [i[0] for i in h]

    # Return the Name of the movies
    return df1['Movie Name'].iloc[result1].drop_duplicates()


print()
user_id = int(input("Enter your user id: "))
print()
print("Top 5 movies recommended or you: ")
print()
print(Recommend(user_id))

# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies.insert(len(movies.columns),'tokens',[tokenize_string(g) for g in movies['genres']])
    #print(movies)
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies= tokenize(movies)
    >>> movies,vocab= featurize(movies)
    >>> print(vocab)
    {'sci-fi': 2, 'romance': 1, 'horror': 0}
    >>> for matrix in movies['features']: print(matrix.toarray())
    [[ 0.30103  0.30103  0.     ]]
    [[ 0.       0.       0.30103]]

    """
    ###TODO
    #the list of csr_matrix
    feature_list = []
    
    #creating  a list of counter_dict (one for each doc/movie)
    doc_c = [] #for finding freq later
    c =Counter() #to get voacb and num_features and df(i)
    for t in movies['tokens']:
        x =Counter(t)
        doc_c.append(x)
        c.update(list(x.keys()))
        
    #creating vocab
    vocab = {term:index for index,term in enumerate(sorted(list(c)))}
    num_features =len(vocab)
    
    #prepare feature_list
    N=len(doc_c)
    
    for c in doc_c:
        data = []
        row_ind = []
        col_ind = []
        for term,freq in c.items():
            row_ind.append(0)
            col_ind.append(vocab[term])
            data.append(freq/max(c.values()) * math.log10(N/c[term]))
        feature_list.append(csr_matrix((data,(row_ind,col_ind)),shape=(1,num_features)))
        
    #insearting feature_list in movies 
    movies.insert(len(movies.columns),'features',feature_list)
    return movies,vocab

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    
    >>> cosine_sim(csr_matrix([1,2,0]), csr_matrix([4,-5,6]))
    
    """
    ###TODO
    return a.dot(b.transpose()).toarray()[0][0] / (np.linalg.norm(a.toarray())*np.linalg.norm(b.toarray()))
    
def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
      
    >>> movies = pd.DataFrame([[123, 'horror|horror|romance|romance|romance', ['horror', 'horror', 'romance', 'romance', 'romance']],\
                                [456, 'comedy|horror', ['comedy', 'horror']],\
                                [789, 'horror', ['horror']],\
                                [000, 'action', ['action']]],\
                               columns=['movieId', 'genres', 'tokens'])
    >>> movies, vocab = featurize(movies)
    >>> ratings_train = pd.DataFrame([\
                 [9, 123, 2.5, 1260759144],\
                 [9, 456, 3.5, 1260759144],\
                 [9, 789, 1, 1260759144],\
                 [8, 123, 4.5, 1260759144],\
                 [8, 456, 4, 1260759144],\
                 [8, 789, 5, 1260759144],\
                 [7, 123, 2, 1260759144],\
                 [7, 456, 3, 1260759144]],\
                                      columns=['userId', 'movieId', 'rating', 'timestamp'])
    >>> ratings_test = pd.DataFrame([\
                 [7, 789, 4, 1260759144]],\
                                     columns=['userId', 'movieId', 'rating', 'timestamp'])
    >>> make_predictions(movies, ratings_train, ratings_test)    
                             
    """
    ###TODO
    predicted = []
    for test_index,test_row in ratings_test.iterrows():     
        #the p+ve similatiry store weighted ratings while n-ve only ratings
        weights = {'p_ratings':[],'p_weights':[],'n':[]}
        feature_test = movies[movies.movieId==test_row['movieId']].get_values()[0][4]
        for train_index,train_row in ratings_train[ratings_train.userId==test_row['userId']].iterrows():
            feature_train = movies[movies.movieId==train_row['movieId']].get_values()[0][4]
            #find cosine/weight
            sim_m=cosine_sim(feature_test,feature_train)
            if sim_m>0:
                weights['p_ratings'].append(train_row['rating'])
                weights['p_weights'].append(sim_m)
            else:
                weights['n'].append(train_row['rating'])
        if len(weights['p_ratings'])!=0 : 
            #weighted average
            predicted.append(np.dot(np.array(weights['p_ratings']),np.array(weights['p_weights'])) /sum(weights['p_weights']))
        else :
            predicted.append(np.mean(weights['n']))
    return np.array(predicted)
        

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])
    
if __name__ == '__main__':
    main()

import numpy as np
# import tensorflow as tf
import torch
from matplotlib import pyplot as plt
import os
import pandas as pd
import re
import sentence2vec
from sklearn.manifold import TSNE

class Get_Data():
    

    def get_movie_message(self, path = './movies.dat'):
        """
        Read Movie DataSets
        """
        ''' read file'''
        new_movies = pd.DataFrame(columns=['MovieID', 'Title', 'Genres', 'Year'])
        movies_title = ['MovieID', 'Title_Year', 'Genres']
        movies = pd.read_csv(path, sep='::', header=None, names=movies_title, engine = 'python')

        ''' map Genres into number 0, 1, 2...'''
        movie_genres_map = {
            "Action" : 0,
            "Adventure" : 1,
            "Animation" : 2,
            "Children's" : 3,
            "Comedy" : 4,
            "Crime" : 5,
            "Documentary" : 6,
            "Drama" : 7,
            "Fantasy" : 8,
            "Film-Noir" : 9,
            "Horror" : 10,
            "Musical" : 11,
            "Mystery" : 12,
            "Romance" : 13,
            "Sci-Fi" : 14,
            "Thriller" : 15,
            "War" : 16,
            "Western" : 17
        }

        pattern = re.compile(r'\((\d+)\)$')
        ''' seperate title and year'''
        Title_list = []
        for index, row in movies.iterrows():
            Title = re.sub(pattern, "", row['Title_Year'])
            Title_list.append(Title)
            Year = row['Title_Year'].replace(Title, '')
            Year = Year.strip('(')
            Year = Year.strip(')')
            Year = int(Year)
            new = pd.DataFrame(
                {
                    'MovieID': row['MovieID'],
                    'Title':Title,
                    'Genres': row['Genres'],
                    'Year': Year,
                },
                index=[1]
            ) 
            new_movies=new_movies.append(new,ignore_index=True)

        ''' convert title into array by TfidfVectorizer and t-SNE'''
        if not os.path.exists("title_array.npy"):
            s2v = sentence2vec.SentenceToVector()
            title_embedded = s2v.model(Title_list)
        
        ''' dataframe -> dictionary'''
        movie_list = new_movies.to_dict('records')
        # print(title_embedded.shape)
        # s2v.plotarr(title_embedded)
        # make Genres into list-vector
        title_array = np.load("title_array.npy")
        i = 0
        for movie in movie_list:
            Genres = movie['Genres']
            Genres_list_str = Genres.split('|')
            Genres_list_int = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for Genres_str in Genres_list_str:
                Genres_list_int[movie_genres_map[Genres_str]] = 1
            movie['Genres'] = Genres_list_int
            movie['Title'] = title_array[i, :]
            i += 1
        # print(movie_list)
        self.movies_data = movie_list
        return new_movies

    def get_user_message(self, path = './users.dat'):
        """
        Read User Datasets
        """
        ''' read file'''
        user_title = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        users = pd.read_csv(path, sep='::', header=None, names=user_title, engine = 'python')
        ''' map Sex to 0, 1'''
        gender_map = {'F':0, 'M':1}
        users['Gender'] = users['Gender'].map(gender_map)
        ''' map Age to number 0, 1, 2...'''
        age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
        users['Age'] = users['Age'].map(age_map)
        ''' delete Zip-code'''
        users = users.drop('Zip-code', axis = 1)
        users_list = users.to_dict('records')
            
        self.users_data = users_list
        return users_list



if __name__ == "__main__":
    movie = Get_Data()
    movie.get_movie_message()
    print(movie.movies_data)
    movie.get_user_message()
    print(movie.users_data)
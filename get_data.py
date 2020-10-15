import numpy as np
# import tensorflow as tf
import torch
from matplotlib import pyplot as plt
import os
import pandas as pd
import re

class Get_Data():
    

    def get_movie_message(self, path = './movies.dat'):
        """
        读取Movie数据集
        """
        new_movies = pd.DataFrame(columns=['MovieID', 'Title', 'Genres', 'Year'])
        movies_title = ['MovieID', 'Title_Year', 'Genres']
        movies = pd.read_csv(path, sep='::', header=None, names=movies_title, engine = 'python')
        movies_orig = movies.values
        pattern = re.compile(r'\((\d+)\)$')
        for index, row in movies.iterrows():
            Title = re.sub(pattern, "", row['Title_Year'])
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
        movie_list = new_movies.to_dict('records')
        for movie in movie_list:
            Genres = movie['Genres']
            Genres_list = Genres.split('|')
            movie['Genres'] = Genres_list
        print(movie_list)
        self.movies_data = movie_list
        return new_movies

    def get_user_message(self, path = './users.dat'):
        """
        docstring
        """
        user_title = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        users = pd.read_csv(path, sep='::', header=None, names=user_title, engine = 'python')
        users = users.drop('Zip-code', axis = 1)
        users_list = users.to_dict('records')
        self.users_data = users_list
        return users_list

            
        

movie = Get_Data()
# movie.get_movie_message()
movie.get_user_message()
print(movie.users_data)
# if __name__ == "__main__":
    
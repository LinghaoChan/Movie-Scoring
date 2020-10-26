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

        print("Reading Movie File......")
        
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

        ''' seperate title and year'''
        pattern = re.compile(r'\((\d+)\)$')
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

        ''' loading title_array.npy into array'''
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
        print("Having Read Movie File.")
        return movie_list   #return list(dictionary)

    def get_user_message(self, path = './users.dat'):
        """
        Read User Datasets
        """

        print("Reading User File......")
       
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
        print("Having Read User File.")
        return users_list   #return list(dictionary)

    def get_rating_message(self, path = './ratings.dat'):
        """
        Read Rating Datasets
        """

        print("Reading Rating File......")
       
        ''' read file'''
        rating_title = ['UserID','MovieID', 'ratings', 'timestamps']
        rating_data_df = pd.read_csv(path, sep='::', header=None, names=rating_title, engine = 'python')

        ''' convert dataframe into list(dictionary)'''
        rating_list = rating_data_df.to_dict('records')

        ''' data normalization'''
        MIN_TIMESTAMPS = 1046454590
        MAX_TIMESTAMPS = 956703932
        TIMESTAMPS_SIZE = 89750658
        for rating in rating_list:
            time_normlized = (rating['timestamps'] - MAX_TIMESTAMPS) / TIMESTAMPS_SIZE
            rating['timestamps'] = time_normlized
        # print(rating_list)
        self.rating_data = rating_list
        print("Having Read Rating File.")
        return rating_list

    def merge_movies_users_ratings_data(self):
        """
        Merge movies, users, ratings data
        """

        '''
        get list(dictionary) data
        '''
        pre_movies_data = self.get_movie_message()
        pre_users_data = self.get_user_message()
        pre_ratings_data = self.get_rating_message()

        '''
        list(dictionary) -> dataframe
        '''

        movies_data_to_df = pd.DataFrame.from_records(pre_movies_data)
        users_data_to_df = pd.DataFrame.from_records(pre_users_data)

        '''
        Merge all infomation
        '''
        rating_all_message_list = []
        print("Merging Movies Users Ratings Data")
        index = 0
        for rating in pre_ratings_data:
            '''
            link message
            '''
            UID = rating['UserID']
            MID = rating['MovieID']
            RATING = rating['ratings']
            TIMESTAMPS = rating['timestamps']

            '''
            user message
            '''
            user_row = users_data_to_df[users_data_to_df['UserID'].isin([UID])]
            user_row_list = user_row.to_dict('records')
            user_row_list_item = user_row_list[0]
            GENDER = user_row_list_item['Gender']
            AGE = user_row_list_item['Age']
            OCCUPTION = user_row_list_item['Occupation']

            '''
            movie message
            '''
            movie_row = movies_data_to_df[movies_data_to_df['MovieID'].isin([MID])]
            movie_row_list = movie_row.to_dict('records')
            movie_row_list_item = movie_row_list[0]
            TITLE = movie_row_list_item['Title']
            GENRES = movie_row_list_item['Genres']
            YEAR = movie_row_list_item['Year']
             
            '''
            get a list(dictionary)
            '''
            rating_dictionary = {
                'UserID' : UID, 
                'Gender' : GENDER, 
                'Age' : AGE, 
                'Occupation' : OCCUPTION, 
                'MovieID' : MID, 
                'Title' : TITLE, 
                'Genres' : GENRES, 
                'Year' : YEAR, 
                'Timestamps' : TIMESTAMPS, 
                'Ratings' : RATING
            }
            if index % 10000 == 0:
                print(index)
            index += 1

            rating_all_message_list.append(rating_dictionary)
            # print(rating_dictionary)
        '''
        list(dictionary) -> dataframe and save "merge_movies_users_ratings_data.json"
        '''        
        rating_all_message_df = pd.DataFrame.from_records(rating_all_message_list)
        rating_all_message_df.to_csv("merge_movies_users_ratings_data.csv")
        rating_all_message_df.to_csv("merge_movies_users_ratings_data.dat")
        print("Having Merged Movies Users Ratings Data")
        print(rating_all_message_df)
            
        # ratings_data_to_df = pd.DataFrame.from_records(pre_ratings_data)
        
        # print(movies_data_to_df, users_data_to_df)


if __name__ == "__main__":
    data = Get_Data()
    # data.get_movie_message()
    # print(data.movies_data)
    # data.get_user_message()
    # print(data.users_data)
    # data.get_rating_message()
    # print(data.rating_data)
    data.merge_movies_users_ratings_data()
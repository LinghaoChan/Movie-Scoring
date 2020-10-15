from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

class SentenceToVector():

    def model(self, text_list):
        """
        convert sentence into vector
        """
        cv=TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
        vec=cv.fit_transform(text_list)
        arr=vec.toarray()
        print("preparing for tsne...")
        arr_embedded = TSNE(n_components=3).fit_transform(arr)
        print("finished for tsne...")
        np.save("title_array.npy",arr_embedded) 
        print("tsne saved")
        # plotarr(arr)
        return arr

    def plotarr(self, arr):
        """
        print array's first 2 dimension
        """
        x = arr[: , 0]
        y = arr[: , 1]
        plt.scatter(x, y)
        plt.show()
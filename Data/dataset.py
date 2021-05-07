import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder


class MushroomData():
    def __init__(self):
        location = os.path.join(os.path.dirname(__file__), 'mushrooms.csv')
        self.x, self.y = self.assemble_matrix(location)

    def assemble_matrix(self, file_name):
        data = pd.read_csv(file_name)
        y = pd.DataFrame(data['class'])
        data.drop('class', axis=1, inplace=True)
        feature_encoder = OneHotEncoder(drop='if_binary')
        x = feature_encoder.fit_transform(data).toarray()
        y = feature_encoder.fit_transform(y).toarray()
        return np.transpose(x), y
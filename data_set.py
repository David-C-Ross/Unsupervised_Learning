import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def prepare_data_set1():
    """
    Prepare the first data set to clustering.
    :return: prepared data
    """
    tracks = pd.read_csv("~/Downloads/fma_metadata/tracks.csv",low_memory=False,header = [0,1],index_col=0)
    echonest = pd.read_csv("~/Downloads/fma_metadata/echonest.csv",low_memory = False,header = [0,1,2],index_col=0)
    
    tracks.index.rename("Track_ID",inplace=True)
    echonest.index.rename("Track_ID",inplace=True)
    echonest_audio_features = echonest['echonest', 'audio_features']
    
    data = pd.merge(echonest_audio_features,tracks.iloc[:,tracks.columns.get_level_values(1).isin(["genre_top"])],how = "left",left_index = True , right_index = True)
    #renaming the target variable:
    data.rename(columns = {('track', 'genre_top') : "Target"},inplace = True)
    
    nullColumns(data)
    data = dropNull(data, nullColumns)

    le = LabelEncoder()
    le.fit(data['Target'])
    data['Target'] = le.transform(data['Target'])
    
    # scale and normalize the data
    data = featureScaling(data,'Target')
    return data



def prepare_data_set2():
    """
    Prepare the second data set to clustering.
    :return: prepared data
    """
    embedding = np.genfromtxt("graph_embedding.csv", delimiter=',')
    target = np.genfromtxt("graph_target.csv", delimiter=',')
    
    data = pd.DataFrame(embedding)
    data['Target'] = target
    
    le = LabelEncoder()
    le.fit(data['Target'])
    data['Target'] = le.transform(data['Target'])
    
     # scale and normalize the data
    data = featureScaling(data,'Target')
    
    return data
    

def prepare_data_set3():
    """
    Prepare the third data set to clustering.
    :return: prepared data
    """   
    data = pd.read_csv("gas_drift.csv")
    target = pd.read_csv("gas_target.csv")
    data['Target'] = target['0']
    
    le = LabelEncoder()
    le.fit(data['Target'])
    data['Target'] = le.transform(data['Target'])
    
    data = featureScaling(data,'Target')
    
    return data, target['1']
    
    


def split(number_of_data_set):
    data = prepare_data_set(number_of_data_set)

    #splitting the data into train and test
    train, test = train_test_split(data, test_size = 0.3, random_state = 100)

    x_train = train.drop('Target', axis=1)
    y_train = train['Target']
    x_test = test.drop('Target', axis =1)
    y_test = test['Target']
    
    return x_train, y_train, x_test, y_test
    


def featureScaling(dataset, target):
    """
    Scales the data
    :param data: data to scale
    :return: scaled data
    """
    rc = RobustScaler()
    #sc = StandardScaler()
    dataset_scaled = rc.fit_transform(dataset.drop(target, axis=1))
    dataset_scaled = pd.DataFrame(normalize(dataset_scaled), 
                                  index = dataset.index, columns = dataset.drop(target, axis=1).columns)
    dataset_scaled[target] = dataset[target]
    return dataset_scaled


def nullColumns(dataset):
    null_columns = []
    for column in dataset.columns.tolist():
        if(dataset[column].isnull().sum() > 0):
            print(column)
            null_columns = null_columns.append(column)
            
            
def dropNull(dataset, null_list):
    dataset.dropna(axis=0,how='any', inplace = True)
    return dataset


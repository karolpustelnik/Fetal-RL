import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
from collections import Counter
import scipy


def probability_mass(data):
    
    counts = Counter(data) # counting the classes
    total = sum(counts.values()) # total number of classes
    probability_mass = {k:v/total for k,v in counts.items()} # probability mass of the classes
    probability_mass = list(probability_mass.values()) # converting the dictionary to a list
    return probability_mass
    

def train_test_split(data, train_size = 0.9, precision = 0.005):
    """splitting data into train and test sets keeping the same distribution of classes using wasertein's method
    args: data - data frame containing the data
    train_size - size of the train set default
    precision - determines how close the train set size is to the train_size default 0.005 (the smaller the better, but it may take longer to generate sets)"""
    
    print('Splitting data into train and test sets...')
    
    data = pd.read_csv(data) # loading the data
    wass_dist = 1
    videos = list(data['video'].unique()) # list of videos
    train_size = int(train_size * len(videos)) # calculating the number of videos in the train set
    while wass_dist > precision: # while the wasserstein distance is greater than 0.005
        train = random.sample(videos, train_size) # sampling the train set
        test = [x for x in videos if x not in train] # sampling the test set
        train_set = data.loc[data['video'].isin(train)] # creating the train set
        test_set = data.loc[data['video'].isin(test)] # creating the test set
        probability_mass_train = probability_mass(train_set['Class']) # calculating the probability mass of the train set
        probability_mass_test = probability_mass(test_set['Class']) # calculating the probability mass of the test set
        wass_dist = scipy.stats.wasserstein_distance(probability_mass_train, probability_mass_test) # wasserstein distance between distributions
    train_set.to_csv('/data/kpusteln/Fetal-RL/data_preparation/data_biometry/biometry_train.csv', index = False) # saving the train set
    test_set.to_csv('/data/kpusteln/Fetal-RL/data_preparation/data_biometry/biometry_val.csv', index = False) # saving the test set
    
    print('Done!')
    return train_set, test_set
    
def histogram_class_plot(data, title):
    """plotting histogram of the classes"""
    plt.figure(figsize=(10,10))
    plt.hist(data['Class'], bins = 100)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(title + '.png')

def frames_n_merge(frames_path, data_path):
    """
    This function takes the path to the frames and the path to the test/train data and returns merged dataframe.
    """
    frames = pd.read_csv(frames_path)
    data = pd.read_csv(data_path)
    data_merged = pd.merge(data, frames, on='video')
    
    return data_merged

name = 'femur'
data_merged = frames_n_merge('/data/kpusteln/Fetal-RL/data_preparation/outputs/frames_n.csv', 
                             f'/data/kpusteln/fetal/standard_plane/{name}_biometry_clean.csv')

data_merged.to_csv(f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/{name}_biometry_merged.csv', index=False)


train_set, test_set = train_test_split(f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/{name}_biometry_merged.csv')
videos_val = test_set['video'].unique()
videos_val = pd.DataFrame(videos_val, columns=['video'])
videos_val.to_csv(f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/videos_val_{name}.csv', index=False)
videos_train = train_set['video'].unique()
videos_train = pd.DataFrame(videos_train, columns=['video'])
videos_train.to_csv(f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/videos_train_{name}.csv', index=False)

histogram_class_plot(train_set, f'/data/kpusteln/Fetal-RL/data_preparation/plots/Train set distribution {name}')
histogram_class_plot(test_set, f'/data/kpusteln/Fetal-RL/data_preparation/plots/Test set distribution {name}')



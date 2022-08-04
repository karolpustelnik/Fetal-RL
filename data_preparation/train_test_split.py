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
    

def train_test_split(data, train_size = 800, precision = 0.005):
    """splitting data into train and test sets keeping the same distribution of classes using wasertein's method
    args: data - data frame containing the data
    train_size - size of the train set default 800 (there are 898 videos)
    precision - determines how close the train set size is to the train_size default 0.005 (the smaller the better, but it may take longer to generate sets)"""
    
    print('Splitting data into train and test sets...')
    
    data = pd.read_csv(data) # loading the data
    wass_dist = 1
    videos = list(data['video_x'].unique()) # list of videos
    while wass_dist > precision: # while the wasserstein distance is greater than 0.005
        train = random.sample(videos, train_size) # sampling the train set
        test = [x for x in videos if x not in train] # sampling the test set
        train_set = data.loc[data['video_x'].isin(train)] # creating the train set
        test_set = data.loc[data['video_x'].isin(test)] # creating the test set
        probability_mass_train = probability_mass(train_set['Class']) # calculating the probability mass of the train set
        probability_mass_test = probability_mass(test_set['Class']) # calculating the probability mass of the test set
        wass_dist = scipy.stats.wasserstein_distance(probability_mass_train, probability_mass_test) # wasserstein distance between distributions
    train_set.to_csv('./outputs/fetal_extracted_map_train_scr.csv', index = False) # saving the train set
    test_set.to_csv('./outputs/fetal_extracted_map_val_scr.csv', index = False) # saving the test set
    
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


train_set, test_set = train_test_split('./outputs/score_labels_4_cls.csv', 800)
histogram_class_plot(train_set, './outputs/Train set distribution')
histogram_class_plot(test_set, './outputs/Test set distribution')



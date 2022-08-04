import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import cv2
import torch

def labels_correction(labels_path):
    """correcting labels for the dataset
    args: labels_path - path to the labels file
    output: corrected labels: 'labels_corrected.csv'
    new_labels:
    0 - other
    1 - head non-standard plane
    2 - head standard plane
    3 - abdomen non-standard plane
    4 - abdomen standard plane
    5 - femur non standard plane
    6 - femur standard plane"""
    
    print('Correcting labels...')
    labels = pd.read_csv(labels_path) # reading the labels
    labels = labels.drop(axis = 1, labels = ['Symmetrical \nplane',  # dropping the columns
                                    'Thalami', 
                                    'Cavum septi \npellucidi', 
                                    'Cerebellum', 
                                    'Symmetrical\nplane',
                                    'Stomach\nbubble',
                                    'Portal \nsinus',
                                    'Kidneys \nnot visible',
                                    'Both ends \nvisible',
                                    'Angle < 45',
                                    'Comments'])

    #correcting head labels
    heads = labels['Head']
    counter = 0
    for i, head in enumerate(heads):
        sep = '_'
        head = str(head)
        stripped = head.split(sep, 2)
        if len(stripped) > 1:
            if stripped[1] != '1':
                counter += 1
                stripped[1] = '1'
                head = sep.join(stripped)
                heads[i] = head


    #correcting Abdomen labels
    abdomens = labels['Abdomen']
    counter = 0
    for i, abdomen in enumerate(abdomens):
        sep = '_'
        abdomen = str(abdomen)
        stripped = abdomen.split(sep, 2)
        if len(stripped) > 1:
            if stripped[1] != '2':
                counter += 1
                stripped[1] = '2'
                abdomen = sep.join(stripped)
                abdomens[i] = abdomen


    #correcting femur labels
    femurs = labels['Femur']
    counter = 0
    for i, femur in enumerate(femurs):
        sep = '_'
        femur = str(femur)
        stripped = femur.split(sep, 2)
        if len(stripped) > 1:
            if stripped[1] != '3':
                counter += 1
                stripped[1] = '3'
                femur = sep.join(stripped)
                femurs[i] = femur



    labels['Head'] = heads # replacing the old labels with the new ones
    labels['Abdomen'] = abdomens # replacing the old labels with the new ones
    labels['Femur'] = femurs # replacing the old labels with the new ones


    index = pd.concat([labels['Head'].append(labels['Abdomen']).append(labels['Femur'])])

    for i, row in enumerate(index): # creating a new column with the index
        row = str(row)
        row = row.replace('.png', '')
        index.iloc[i] = row


    class_type = pd.concat([labels['Head_class'].append(labels['Abdomen_class']).append(labels['Femur_class'])])

    labels_good = pd.concat([index, class_type], axis = 1) # creating a new dataframe with the new labels

    labels_good.columns = ['index', 'Class']
    labels_good = labels_good.dropna() # dropping the NaN values
    new_column = []
    for i in range(len(labels_good)):
        text = labels_good.iloc[i, 0]
        sep = '_'
        stripped = text.split(sep, 2)[:2]
        text = '_'.join(stripped)
        new_column.append(text)
    labels_good['video'] = new_column
    # creating numerical labels

    labels_good.to_csv('labels_corrected.csv', index = False) # saving the new labels
    
    labels_good = pd.read_csv('labels_corrected.csv') # reading the labels
    for i in range(len(labels_good)):
        if i%10000 == 0:
            print(f'Finished: {int((i/len(labels_good))*100)}%')
        if labels_good.iloc[i][1] == 'Head':
            labels_good.iloc[i][1] = 1
        if labels_good.iloc[i][1] == 'Abdomen':
            labels_good.iloc[i][1] = 3
        if labels_good.iloc[i][1] == 'Femur':
            labels_good.iloc[i][1] = 5
        if labels_good.iloc[i][1] == 'Head non-standard plane' or labels_good.iloc[i][1] == 'Head non-stendard plane':
            labels_good.iloc[i][1] = 1
            
        if labels_good.iloc[i][1] == 'Other' or labels_good.iloc[i][1] == '???' or labels_good.iloc[i][1] == 'Other ' or labels_good.iloc[i][1] == 'nkn':
            labels_good.iloc[i][1] = 0

        if labels_good.iloc[i][1] == 'Head standard plane':
            labels_good.iloc[i][1] = 2

        if labels_good.iloc[i][1] == 'Abdomen non-standard plane' or labels_good.iloc[i][1] == 'Abdomen non-standard place' or labels_good.iloc[i][1] == 'Abomen non-standard plane' or labels_good.iloc[i][1] == ' Abdomen non-standard plane':
            labels_good.iloc[i][1] = 3

        if labels_good.iloc[i][1] == 'Abdomen standard plane' or labels_good.iloc[i][1] == 'Abdomen standard frame' or labels_good.iloc[i][1] == 'Abdomen standard-plane':
            labels_good.iloc[i][1] = 4

        if labels_good.iloc[i][1] == 'Femur non-standard plane':
            labels_good.iloc[i][1] = 5

        if labels_good.iloc[i][1] == 'Femur standard plane':
            labels_good.iloc[i][1] = 6
    index_to_drop = labels_good[labels_good['index'] == '63_2_253'].index # w filmikach nie ma klatki 63_2_253, wiec trzeba usunac ten label
    if len(index_to_drop) > 0:
        labels_good = labels_good.drop(index_to_drop)
    labels_good.to_csv('./outputs/labels_corrected.csv', index = False) # saving the new labels

labels_correction('/data/kpusteln/Fetal-RL/data_preparation/outputs/labele_org.csv')
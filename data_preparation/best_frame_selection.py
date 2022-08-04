import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import torch
from imutils import paths
import argparse
import cv2
from skimage.metrics import structural_similarity

def best_frame_selection(corrected_labels_path, img_path):
    """selecting the best frame for each video
    
    args: corrected_labels_path - path to the corrected labels file
    img_path - path to the images folder
    
    returns: best_frames - dictionary with the best frames for each label saved as 'sharpest_frames_dict.npy'"""
    
    print('Selecting the best frame for each video...')
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    labels = pd.read_csv(corrected_labels_path)

    new_column = []
    for i in range(len(labels)):
        text = labels.iloc[i, 0]
        sep = '_'
        stripped = text.split(sep, 2)[:2]
        text = '_'.join(stripped)
        new_column.append(text)

    labels['video'] = new_column

    labels_key = labels[(labels['Class'] == 6) | (labels['Class'] == 2) | (labels['Class'] == 4)] # selecting the rows with the key labels

    index_vid = labels_key.drop(labels = 'Class', axis = 1) # dropping the class column


    wide = index_vid.pivot(columns = 'video', values = 'index') # pivoting the index column

    best_frames = dict()
    for i in range(len(wide.columns)): 
        print(f'Processing video: {i}/{len(wide.columns)}')
        frames = wide.iloc[:, i]
        frames = frames.dropna()
        video = frames.name
        for frame in frames:
            frame = str(frame)
            image = cv2.imread(img_path + frame + '.png')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            sep = '_'
            stripped = frame.split(sep, 2)[:2]
            text = '_'.join(stripped)
            if text not in best_frames.keys():
                best_frames[text] = {}
            best_frames[text][frame] = fm


    sharpest_frames = []
    for video in best_frames.keys():
        best_frame = sorted(best_frames[video].items(), key=lambda x: x[1], reverse = True)[0][0]
        sharpest_frames.append(best_frame)
        

        
    sharpest_frames_dict = dict()
    for frame in sharpest_frames: # frame is the frame's index
        sep = '_' 
        stripped = frame.split(sep, 2)[:2]
        text = '_'.join(stripped)
        sharpest_frames_dict[text] = frame
        

    np.save("./outputs/sharpest_frames_dict.npy", sharpest_frames_dict) # saving the dictionary to a numpy file
    
best_frame_selection('labels_corrected.csv', '/data/kpusteln/fetal/fetal_extracted/')
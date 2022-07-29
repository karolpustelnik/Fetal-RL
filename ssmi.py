import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity

def ssmi(img_path, labels_path, sharpest_frames_path):
    """selecting the best frame for each label
    args: img_path - path to the images folder
    labels_path - path to the labels file
    
    returns: similarity dictionary - ssmi for each frame in video with reference to the best frame in the video"""
    sharpest_frames_dict = np.load(sharpest_frames_path, allow_pickle = True).item()
    labels = pd.read_csv(labels_path)
    similarity_dict = dict.fromkeys(labels['index'])

    k = 0
    for vid, val in sharpest_frames_dict.items(): #vid is the video name
        videos = labels.loc[labels['video'] == vid] #videos is the video's labels
        ideal_frame = cv2.imread(img_path + str(val) + '.png') #ideal_frame is the ideal frame
        ideal_frame = cv2.cvtColor(ideal_frame, cv2.COLOR_BGR2GRAY)
        for frame_index in videos['index']: #frame_index is the frame's index
            reference_img = cv2.imread(img_path + str(frame_index) + '.png')
            reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            similarity = structural_similarity(reference_img, ideal_frame)
            similarity_dict[frame_index] = similarity
            k += 1
            if k % 100 == 0:
                with open("status.txt", "w") as text_file:
                    text_file.write(f'{k}/{261766}')
    with open("status.txt", "w") as text_file:
                    text_file.write(f'{k}/{261766} done')
    np.save('similarity_dict.npy', similarity_dict)
            
ssmi('/data/kpusteln/fetal/fetal_extracted/', '/data/kpusteln/Fetal-RL/labels_corrected.csv', '/data/kpusteln/Fetal-RL/sharpest_frames_dict.npy')
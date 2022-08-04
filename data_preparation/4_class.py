import pandas as pd



def only_4_class(score_labels_path):
    """ Convert 7 classes to 4 classes. 
    args: score_labels_path: path to the score labels csv file
    0 - other
    1 - head (standard or non-standard plane)
    2 - abdomen (standard or non-standard plane)
    3 - femur (standard or non-standard plane)"""
    
    print('Converting 7 classes to 4 classes...')
    score_labels = pd.read_csv(score_labels_path) # read csv file

    for i in range(len(score_labels)): # for each row
        video = score_labels.iloc[i, 3] # get video name
        sep = '_' # separator
        video = str(video) # convert to string
        stripped = video.split(sep, 2) # split string
        if score_labels.iloc[i, 1] != 0: # if not 0
            score_labels.iloc[i, 1] = stripped[1] # set to second part of string
        
    score_labels.to_csv('./outputs/score_labels_4_cls.csv', index = False) # write to csv file
    print('Done!')
only_4_class('./outputs/score_labels.csv')
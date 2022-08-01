import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

def dist_mean_ssmi(sharpest_franes_dict_path, labels_path, ssmi_data_path):
    from sklearn.preprocessing import MinMaxScaler
    """calculating the mean distance between the frames and the labels
    args: sharpest_franes_dict_path - path to the sharpest frames dictionary
    labels_path - path to the labels file
    ssmi_data_path - path to the ssmi data file
    
    returns:
    saves data frame to 'data_ssmi_dist_cos.csv' a data frame containing mean of ssmi and frames distance (normalized), 
    and cosine similarity between ssmi and frames distance"""
    
    print('calculating...')
    
    
    sharpest_frames = np.load(sharpest_franes_dict_path, allow_pickle = True).item() # loading the sharpest frames dictionary
    dist = dict() 
    distance_list = []
    indexes_list = [] # list of indexes of the frames
    labels = pd.read_csv(labels_path) # loading the labels

    scaler = MinMaxScaler((0, 1)) # creating a scaler to normalize the data
    for vid, val in sharpest_frames.items():
        frames = labels.loc[labels['video'] == vid]
        ideal_frame_full_name = str(val)
        ideal_frame = str(val)
        sep = '_'
        ideal_frame = ideal_frame.split(sep, 2)[-1]
        
        for frame_index in frames['index']:
            reference_frame = str(frame_index)
            sep = '_'
            reference_frame = reference_frame.split(sep, 2)[-1]
            distance = abs(int(reference_frame) - int(ideal_frame))
            indexes_list.append(frame_index)
            distance_list.append(distance)
        distance_list = np.array(distance_list)
        distances = scaler.fit_transform(distance_list.reshape(-1, 1)) * (-1) + 1
        distances = distances.flatten().tolist()
        dictionary = dict(zip(indexes_list, distances))
        dictionary[ideal_frame_full_name] = 1
        dist[vid] = dictionary
        distance_list = []
        indexes_list = []
        
    indexes = []
    distances = []
    for video in dist.keys():
        keys = list(dist[video].keys())
        values = list(dist[video].values())
        indexes.extend(keys)
        distances.extend(values)
        
    dane_dist = pd.DataFrame({'distance': distances}, index = indexes)
    similarity_dict = np.load(ssmi_data_path, allow_pickle = True).item()
    similarity_dict_keys = list(similarity_dict.keys())
    similarity_dict_values = list(similarity_dict.values())
    dane_ssmi = pd.DataFrame({'ssmi_similarity': similarity_dict_values}, index=similarity_dict_keys)
    dane_ssmi = dane_ssmi.dropna()
    data_ssmi_dist = pd.merge(dane_ssmi, dane_dist, left_index=True, right_index=True)

    #CALCULATe cosine similarity (ostatecznie nie uzywany)
    distance_list = []
    indexes_list = []
    from sklearn.preprocessing import MinMaxScaler
    for vid, val in sharpest_frames.items():
        frames = labels.loc[labels['video'] == vid]
        for frame_index in frames['index']:
            reference_frame = str(frame_index)
            reference_dist = data_ssmi_dist['distance'][reference_frame]
            reference_ssmi = data_ssmi_dist['ssmi_similarity'][reference_frame]
            cosine_sim = 1 - cosine([1,1], [reference_dist, reference_ssmi])
            indexes_list.append(frame_index)
            distance_list.append(cosine_sim)
            
    data_cos = pd.DataFrame({'cosine_sim': distance_list}, index=indexes_list)
    data_ssmi_dist_cos = pd.merge(data_ssmi_dist, data_cos, left_index=True, right_index=True)
    mean_dist_ssmi = data_ssmi_dist.mean(axis = 1) #mean distance + ssmi

    data_ssmi_dist['mean_dist_ssmi'] = mean_dist_ssmi
    ##normalizing mean_dist_ssmi
    new_column = []
    for i in range(len(data_ssmi_dist)):
        text = data_ssmi_dist.index[i]
        sep = '_'
        stripped = text.split(sep, 2)[:2]
        text = '_'.join(stripped)
        new_column.append(text)
    data_ssmi_dist['video'] = new_column
    data_ssmi_dist['index'] = data_ssmi_dist.index
    
    wide =  data_ssmi_dist.pivot(columns = 'video', values = 'mean_dist_ssmi', index = 'index') # pivoting the index column
    idxs = []
    values = []
    for col_name in wide.columns:
        column = wide[col_name]
        column = column.dropna()
        idxs.extend(list(column.index))
        column = np.array(column).reshape(-1, 1)
        column = scaler.fit_transform(column)
        column = column.reshape(-1)
        values.extend(column)
    df = pd.DataFrame({'index': idxs, 'score': values})
    data_ssmi_dist = pd.merge(data_ssmi_dist, df, on = 'index')
    labels_corrected = pd.read_csv('labels_corrected.csv')
    labels_corrected.set_index('index', inplace = True)
    data_ssmi_dist.set_index('index', inplace = True)
    data_ssmi_dist = pd.merge(data_ssmi_dist, labels_corrected, left_index=True, right_index=True)
    data_ssmi_dist['index'] = data_ssmi_dist.index
    score_labels = data_ssmi_dist[['index', 'Class','score', 'video_x']]
    score_labels.to_csv('score_labels.csv', index = False)
    data_ssmi_dist.to_csv('data_ssmi_dist.csv', index = False)

dist_mean_ssmi('sharpest_frames_dict.npy', 'labels_corrected.csv', 'similarity_dict.npy')
        
            
            
        
        

        
            
            
        
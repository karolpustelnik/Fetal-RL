import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
#CALCULATe distances
sharpest_frames = np.load('sharpest_frames_dict.npy', allow_pickle = True).item()
dist = dict()
distance_list = []
indexes_list = []
labels = pd.read_csv('labels_corrected.csv')


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
    distances = (MinMaxScaler((0, 1)).fit_transform(distance_list.reshape(-1, 1)) * (-1)) + 1
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
similarity_dict = np.load('similarity_dict.npy', allow_pickle = True).item()
similarity_dict_keys = list(similarity_dict.keys())
similarity_dict_values = list(similarity_dict.values())
dane_ssmi = pd.DataFrame({'ssmi_similarity': similarity_dict_values}, index=similarity_dict_keys)
dane_ssmi = dane_ssmi.dropna()
data_ssmi_dist = pd.merge(dane_ssmi, dane_dist, left_index=True, right_index=True)

#CALCULATe cosine similarity
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
wide =  data_ssmi_dist.pivot(columns = 'video', values = 'mean_dist_ssmi', index = 'index') # pivoting the index column
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
data_frame = pd.DataFrame()
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
df = pd.DataFrame({'index': idxs, 'mean_dist_ssmi_normalized': values})
df = pd.merge(df, data_ssmi_dist, on = 'index')

data_ssmi_dist.to_csv('data_ssmi_dist.csv', index = True)

data_ssmi_dist_cos.to_csv('data_ssmi_dist_cos.csv', index = True) #save data

    
        
        
    
    

    
        
        
    
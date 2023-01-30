import pandas as pd

def frames_n_test(frames_path, data_path):
    """
    This function takes the path to the frames and the path to the test data and returns merged dataframe.
    """
    frames = pd.read_csv(frames_path)
    data = pd.read_csv(data_path)
    data_merged = pd.merge(data, frames, on='video')
    
    return data_merged

data_merged = frames_n_test('./outputs/frames_n.csv', './outputs/biometry_val.csv')

data_merged.to_csv('./outputs/biometry_val_final.csv', index=False)
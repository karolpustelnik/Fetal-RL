import numpy as np
import pandas as pd



data = pd.read_csv('./outputs/labels_corrected.csv')
frames_n = data['video'].value_counts()
frames_n = pd.DataFrame({'video': frames_n.index, 'frames_n': frames_n.values})
frames_n.to_csv('/data/kpusteln/Fetal-RL/data_preparation/outputs/frames_n.csv', index=False)























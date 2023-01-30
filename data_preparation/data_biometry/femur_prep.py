import pandas as pd

import pandas as pd
import numpy as np
import torch


## PREPARATION OF THE DATA
biometry = pd.read_csv('/data/kpusteln/Fetal-RL/data_preparation/outputs/biometry_org.csv')

femur_biometry = biometry[['ID', 'FL', 'femur_ps']]
abdomen_biometry = biometry[['ID', 'AC', 'abdomen_ps']]
head_biometry = biometry[['ID', 'HC', 'head_ps']]

femur_biometry['ID'] = femur_biometry['ID'].astype(str)
abdomen_biometry['ID'] = abdomen_biometry['ID'].astype(str)
head_biometry['ID'] = head_biometry['ID'].astype(str)

for i in range(len(femur_biometry)):
    idx = femur_biometry.loc[i, 'ID']
    new_idx = idx + '_3'
    femur_biometry.loc[i, 'ID'] = new_idx
    
for i in range(len(abdomen_biometry)):
    idx = abdomen_biometry.loc[i, 'ID']
    new_idx = idx + '_2'
    abdomen_biometry.loc[i, 'ID'] = new_idx
    
for i in range(len(head_biometry)):
    idx = head_biometry.loc[i, 'ID']
    new_idx = idx + '_1'
    head_biometry.loc[i, 'ID'] = new_idx

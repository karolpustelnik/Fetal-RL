import pandas as pd
import numpy as np
import torch


## PREPARATION OF THE DATA
biometry = pd.read_csv('/data/kpusteln/Fetal-RL/data_preparation/outputs/biometry_org.csv')

femur_biometry = biometry[['ID', 'FL', 'femur_ps']]
femur_biometry['ID'] = femur_biometry['ID'].astype(str)

for i in range(len(femur_biometry)):
    idx = femur_biometry.loc[i, 'ID']
    new_idx = idx + '_3'
    femur_biometry.loc[i, 'ID'] = new_idx
    
femur_data = pd.read_csv('/data/kpusteln/Fetal-RL/data_preparation/outputs/femur_data.csv')
femur_merged = pd.merge(femur_data, femur_biometry, left_on='video_x', right_on='ID')
femur_merged = femur_merged.rename(columns={'video_x': 'video'})

femur_merged.drop(['ID', 'score'], axis=1, inplace=True)

femur_merged.to_csv('/data/kpusteln/Fetal-RL/data_preparation/outputs/femur_merged.csv', index=False)

## DATA LOADER
import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision.transforms as transforms
from torchvision.utils import save_image

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class Fetal(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = pd.read_csv(self.ann_path)

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database.iloc[index]

        # images
        images = self._load_image(self.data_path  + idb[0] + '.png')
        if self.transform is not None:
            images = self.transform(images)

        # target
        indexes = idb[0]
        target = idb[1]
        #score = idb[2]
        score = 1
        video = idb[2]
        femur_len = idb[3]
        pixel_spacing = idb[4]
        if self.target_transform is not None:
            target = self.target_transform(target)
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        return images, target, indexes, pixel_spacing, video, femur_len

    def __len__(self):
        return len(self.database)


t = transforms.Compose([transforms.ToTensor(),])

fetal = Fetal(root = '/data/kpusteln/fetal/fetal_extracted/', ann_path = './outputs/femur_merged.csv', transform=t)

batch_size = 1

loader = torch.utils.data.DataLoader(
      fetal,
      batch_size=batch_size,
      shuffle=False,
      num_workers=40,
  )

indexes_list = []
height_list = []
width_list = []
labels = []
pixel_spacings = []
videos = []
femur_lens = []
i = 0
px = 0.15
print('Calculating pixel spacing...')
for img, label, index, pixel_spacing, video, femur_len in loader:
    height = img.shape[2]
    width = img.shape[3]
    new_height = height*(pixel_spacing/px)
    new_width = width*(pixel_spacing/px)
    label = label.numpy()
    height_list.extend(new_height.numpy())
    width_list.extend(new_width.numpy())
    indexes_list.extend(index)
    labels.extend(label)
    femur_lens.extend(femur_len.numpy())
    videos.extend(video)
    pixel_spacings.extend(pixel_spacing.numpy())
    i+=1
    if i%1000 == 0:
        print(f'finished: {int(i/len(loader)*100)}%')   
    
femur_bioemtry_new = pd.DataFrame({'ID': indexes_list, 'height': height_list, 'width': width_list, 'label': labels, 'video': videos, 'femur_len': femur_lens, 'pixel_spacing': pixel_spacings})
femur_bioemtry_new.to_csv('./outputs/femur_bioemtry_new.csv', index=False)
print('Finished calculating pixel spacing!')



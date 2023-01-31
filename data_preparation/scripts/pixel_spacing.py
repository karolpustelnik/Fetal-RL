import pandas as pd
import numpy as np
import torch


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
        #index,Class,video,measure,ps,frames_n,measure_scaled,days,frame_loc
        # images
        images = self._load_image(self.data_path  + idb[0] + '.png')
        if self.transform is not None:
            images = self.transform(images)

        # target
        indexes = idb[0]
        target = idb[1]
        video = idb[2]
        measure = idb[3]
        pixel_spacing = idb[4]
        length = idb[5]
        measure_scaled = idb[6]
        days = idb[7]
        frame_loc = idb[8]
        if self.target_transform is not None:
            target = self.target_transform(target)
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        return images, target, indexes, video, measure, pixel_spacing, length, measure_scaled, days, frame_loc

    def __len__(self):
        return len(self.database)


t = transforms.Compose([transforms.ToTensor(),])

fetal = Fetal(root = '/data/kpusteln/fetal/fetal_extracted/', 
              ann_path = f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_val.csv', 
              transform=t)

batch_size = 1

loader = torch.utils.data.DataLoader(
      fetal,
      batch_size=batch_size,
      shuffle=False,
      num_workers=40,
  )

labels = []
indexes = []
videos = []
measures = []
pixel_spacings = []
lens = []
measure_scaled_list = []
days_list = []
frame_loc_list = []
heights = []
widths = []
heights_org = []
widths_org = []
i = 0

print('Calculating pixel spacing...')
for img, label, index, video, measure, pixel_spacing, length, measure_scaled, days, frame_loc in loader:
    height = img.shape[2]
    width = img.shape[3]
    pixel_spacing = pixel_spacing.numpy()
    widths_org.append(width)
    heights_org.append(height)
    widths.extend(width * pixel_spacing)
    heights.extend(height * pixel_spacing)
    scale = width/512 #img size 512
    pixel_spacing = pixel_spacing/scale
    label = label.numpy()
    measure = measure.numpy()
    length = length.numpy()
    measure_scaled = measure_scaled.numpy()
    days = days.numpy()
    frame_loc = frame_loc.numpy()
    labels.extend(label)
    indexes.extend(index)
    videos.extend(video)
    measures.extend(measure)
    pixel_spacings.extend(pixel_spacing)
    lens.extend(length)
    measure_scaled_list.extend(measure_scaled)
    days_list.extend(days)
    frame_loc_list.extend(frame_loc)
    i+=1
    if i%1000 == 0:
        print(f'finished: {int(i/len(loader)*100)}%')   
    
femur_bioemtry_new = pd.DataFrame({'index': indexes, 'Class': labels, 'video': videos, 'measures': measures, 
                                   'ps': pixel_spacings, 'frames_n': lens, 'measure_scaled': measure_scaled_list, 
                                   'days': days_list, 'frame_loc': frame_loc_list, 'height': heights, 'width': widths,
                                   'height_org': heights_org, 'width_org': widths_org})
femur_bioemtry_new.to_csv('/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/biometry_val_scaled_size.csv', index=False)
print('Finished calculating pixel spacing!')



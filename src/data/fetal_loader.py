import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
import cv2
import albumentations as A
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

"""labels:
0 - other
1 - head non-standard plane
2 - head standard plane
3 - abdomen non-standard plane
4 - abdomen standard plane
5 - femur non standard plane
6 - femur standard plane
"""

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)



class Fetal_frame_eval_cls(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)

    def _load_image(self, path):
        try:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except:
            print("ERROR IMG NOT LOADED: ", path)
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
        frame_idx = idb[0]
        video = idb[1]
        ps = idb[2]
        Class = 1

        images = self._load_image(self.data_path  + frame_idx + '.png')
        images = np.expand_dims(images, 2)
        t = transforms.Compose([transforms.ToTensor(),
        transforms.Resize((450, 600)),
        transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=0.1354949, std=0.18222201)])
        images = t(images)
        if self.transform is not None:
            images = self.transform(images)
        
        return images, frame_idx, video, ps, Class

    def __len__(self):
        return len(self.database)
    
    def load_video(self, video):
        indexes = self.database.query('video == @video')['index']
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in indexes:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(index)
        return torch.stack(images), torch.stack(Classes), torch.stack(measures), torch.stack(pss), torch.stack(frames_n), torch.stack(measures_normalized), torch.stack(indexes)
    
    def load_batch(self, index_list):
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in index_list:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(index)
        return torch.stack(images), torch.stack(Classes), torch.stack(measures), torch.stack(pss), torch.stack(frames_n), torch.stack(measures_normalized), torch.stack(indexes)
                      
    
    
class Fetal_frame_eval_reg(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)

    def _load_image(self, path):
        try:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except:
            print("ERROR IMG NOT LOADED: ", path)
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
        frame_idx = idb[0]
        ps = idb[3]
        video = idb[1]
        Class = idb[2]
        #ps = torch.tensor(idb[4])

        images = self._load_image(self.data_path  + frame_idx + '.png')
        images = np.expand_dims(images, 2)
        t = transforms.Compose([transforms.ToTensor(),
        transforms.Resize((450, 600)),
        transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=0.1354949, std=0.18222201)])
        images = t(images)
        if self.transform is not None:
            images = self.transform(images)
        
        return images, frame_idx, video, ps, Class

    def __len__(self):
        return len(self.database)
    
    def load_video(self, video):
        indexes = self.database.query('video == @video')['index']
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in indexes:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(index)
        return torch.stack(images), torch.stack(Classes), torch.stack(measures), torch.stack(pss), torch.stack(frames_n), torch.stack(measures_normalized), torch.stack(indexes)
    
    def load_batch(self, index_list):
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in index_list:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(index)
        return torch.stack(images), torch.stack(Classes), torch.stack(measures), torch.stack(pss), torch.stack(frames_n), torch.stack(measures_normalized), torch.stack(indexes)
                      
    
    
        
class Fetal_frame(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None, img_scaling = False):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)
        self.img_scaling = img_scaling
        

    def _load_image(self, path):
        try:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except:
            print("ERROR IMG NOT LOADED: ", path)
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
        frame_idx = idb[0]
        Class = torch.tensor(idb[1])
        video = idb[2]
        measure = torch.tensor(idb[3])
        ps = torch.tensor(idb[4])
        frames_n = torch.tensor(idb[5])
        measure_scaled = torch.tensor(idb[6], dtype=torch.float32)
        days_normalized = torch.tensor(idb[7], dtype=torch.float32)
        frame_loc = torch.tensor(idb[8], dtype=torch.float32)
        height = torch.tensor(idb[9])
        width = torch.tensor(idb[10])
        measure_normalized = torch.tensor(idb[13], dtype=torch.float32)
        images = self._load_image(self.data_path  + frame_idx + '.png')
        
        new_height = clamp(int(height*2.161191086437513), 0, 512)
        new_width = clamp(int(width*2.161191086437513), 0, 512)
        if self.img_scaling:
            padding = A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, mask_value=0, always_apply=False, p=1.0)
            rescale = A.Resize(new_height, new_width)
            images = rescale(image = images)['image']
            images = padding(image=images)['image']
            images = np.expand_dims(images, 2)
            t = transforms.Compose([transforms.ToTensor(),])
            images = t(images)
        else:
            images = np.expand_dims(images, 2)
            t = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((450, 600)),
            transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=0.1354949, std=0.18222201)])
            images = t(images)
        if self.transform is not None:
            images = self.transform(images)

        return images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized

    def __len__(self):
        return len(self.database)
    
    def load_video(self, video):
        indexes = self.database.query('video == @video')['index']
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in indexes:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(index)
        return torch.stack(images), torch.stack(Classes), torch.stack(measures), torch.stack(pss), torch.stack(frames_n), torch.stack(measures_normalized), torch.stack(indexes)
    
    def load_batch(self, index_list):
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        for i in index_list:
            image, Class, measure, ps, frame_n, measure_normalized, index = self.__getitem__(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures_normalized.append(measure_normalized)
            indexes.append(torch.tensor(index))
        images = torch.stack(images)
        Classes = torch.stack(Classes)
        measures = torch.stack(measures)
        pss = torch.stack(pss)
        frame_n = torch.stack(frames_n)
        measure_normalized = torch.stack(measures_normalized)
        indexes = torch.stack(indexes)
        return images, Classes, measures, pss, frame_n, measure_normalized, indexes
                      
    
class Video_Loader():
    def __init__(self, root, videos_path, ann_path, transform=None, target_transform=None, img_scaling = False):
        
        self.data_path = root
        self.ann_path = ann_path
        self.videos = pd.read_csv(videos_path)
        self.data = pd.read_csv(self.ann_path)
        self.img_scaling = img_scaling
        
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
    
    def __len__(self):
        return len(self.data)
    
    
    def _load_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.data.iloc[index]
        frame_idx = idb[0]
        Class = torch.tensor(idb[1])
        video = idb[2]
        measure = torch.tensor(idb[3])
        ps = torch.tensor(idb[4])
        frames_n = torch.tensor(idb[5])
        measure_scaled = torch.tensor(idb[6], dtype=torch.float32)
        days_normalized = torch.tensor(idb[7], dtype=torch.float32)
        frame_loc = torch.tensor(idb[8], dtype=torch.float32)
        height = torch.tensor(idb[9])
        width = torch.tensor(idb[10])
        measure_normalized = torch.tensor(idb[13], dtype=torch.float32)
        images = self._load_image(self.data_path  + frame_idx + '.png')
        
        new_height = clamp(int(height*2.161191086437513), 0, 512)
        new_width = clamp(int(width*2.161191086437513), 0, 512)
        if self.img_scaling:
            padding = A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, mask_value=0, always_apply=False, p=1.0)
            rescale = A.Resize(new_height, new_width)
            images = rescale(image = images)['image']
            images = padding(image=images)['image']
            images = np.expand_dims(images, 2)
            t = transforms.Compose([transforms.ToTensor(),])
            images = t(images)
        else:
            images = np.expand_dims(images, 2)
            t = transforms.Compose([transforms.ToTensor(),
            transforms.Resize((450, 600)),
            transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=0.1354949, std=0.18222201)])
            images = t(images)
        if self.transform is not None:
            images = self.transform(images)

        return images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized
    
    
    def __getitem__(self, index):
        video = self.videos[index]
        batch = self.data[self.data['video'] == video].sample(8)
        indexes = batch['index']
        images = list()
        Classes = list()
        measures = list()
        pss = list()
        frames_n = list()
        measures_normalized = list()
        indexes = list()
        days = list()
        frames_loc = list()
        for i in indexes:
            image, Class, measure, ps, frame_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized = self._load_item(i)
            images.append(image)
            Classes.append(Class)
            measures.append(measure)
            pss.append(ps)
            frames_n.append(frame_n)
            measures.append(measure_scaled)
            indexes.append(index)
            days.append(days_normalized)
            frames_loc.append(frame_loc)
            measures_normalized.append(measure_normalized)
        return images, Class, measures, ps, frames_n, measure_scaled, index, days, frame_loc, measure_normalized
    

class Fetal_vid_new(data.Dataset):
    def __init__(self, videos_path, root, ann_path, transform=None, target_transform=None):
        
        
        self.videos = pd.read_csv(videos_path)
        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("!!ERROR IMG NOT LOADED!!: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, frame_positions, labels)
        """
        #index, Class, video, FL, femur_ps, frames_n
        images = list() # list of images
        vid = self.videos.iloc[index][0]
        vid_len = self.database.query('video == @vid')['frames_n'].iloc[0]
        Class = self.database.query('video == @vid')['Class'].iloc[0]
        part = 'head_ps' if Class == 2 else 'abdomen_ps' if Class == 4 else 'femur_ps'
        ps = self.database.query('video == @vid')['abdomen_ps'].iloc[0]
        indexes = self.database.query('video == @vid')['index']
        length = 'HC' if Class == 2 else 'AC' if Class == 4 else 'FL'
        measure = self.database.query('video == @vid')['length'].iloc[0]
        measure_scaled = measure/ps
        # Max measure scaled: head = 214.14944514917548
        # max measure scaled: abdomen = 217.2456030876609
        # max measure scaled: femur = 72.1250937626219
        max_measure = 214.14944514917548 if Class == 2 else 217.2456030876609 if Class == 4 else 72.1250937626219
        measure_normalized = measure_scaled/max_measure
        #print(self.database.query(vid[0]))
        #index, #class, #video, #frames_n, abdomen_ps, AC
        # images
        buckets = round(vid_len/8)
        indexes.index = [i for i in range(vid_len)]
        
        for frame in range(0, vid_len, buckets):
            #print(frame)
            frame_idx = indexes.iloc[frame]
            image = self._load_image(self.data_path  + frame_idx + '.png')
            # transform image              
            if self.transform is not None:
                image = self.transform(image)
            images.append(image) # append image to list


        # target
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        images = images[:8]
        images = torch.stack(images)
        images = images.permute(1, 0, 2, 3)
        frames_position = [i+1 for i in range(8)]
        frames_position = torch.tensor(frames_position)
        
        return images, index, Class, vid, measure, ps, vid_len, measure_normalized

    def __len__(self):
        return len(self.videos)


class Fetal_vid_old(data.Dataset):
    def __init__(self, videos_path, root, ann_path, transform=None, target_transform=None):
        
        
        self.videos = pd.read_csv(videos_path)
        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("ERROR IMG NOT LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, frame_positions, labels)
        """
        #index, Class, video, FL, femur_ps, frames_n
        images = list() # list of images
        frames_classes = list()
        vid = self.videos.iloc[index][0]
        vid_len = self.database.query('video == @vid')['frames_n'].iloc[0]
        Classes = self.database.query('video == @vid')['Class']
        #part = 'head_ps' if Class == 2 else 'ps' if Class == 4 else 'femur_ps'
        ps = self.database.query('video == @vid')['ps'].iloc[0]
        indexes = self.database.query('video == @vid')['index']
        #length = 'HC' if Class == 2 else 'AC' if Class == 4 else 'FL'
        measure = self.database.query('video == @vid')['measure'].iloc[0]
        measure_normalized = self.database.query('video == @vid')['measure_scaled'].iloc[0]
        
        for frame in range(0, vid_len):
            #print(frame)
            frame_idx = indexes.iloc[frame]
            frame_class = Classes.iloc[frame]
            image = self._load_image(self.data_path  + frame_idx + '.png')
            # transform image              
            if self.transform is not None:
                image = self.transform(image)
            images.append(image) # append image to list
            frames_classes.append(frame_class)

            
            

        # target
        frames_position = [i+1 for i in range(vid_len)]
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        images = torch.stack(images)
        frames_classes = torch.tensor(frames_classes)
        images = images.permute(1, 0, 2, 3)
        frames_position = torch.tensor(frames_position)
        #cut to max 400 frames
        images = images[:, :64, :, :]
        frames_classes = frames_classes[:64]
        print(f'shape of frames_classes: {frames_classes.shape}')

        return images, index, frames_classes, vid, measure, ps, vid_len, measure_normalized

    def __len__(self):
        return len(self.videos)
    
    
# fetal_frame = Fetal_frame('/data/kpusteln/fetal/fetal_extracted/', 
#                         '/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_train.csv', 
#                         transform = transforms.ToTensor())


# buffer = torch.tensor([], dtype=torch.int)
# labels = [1, 2, 3, 4, 5, 6, 7, 8, 6, 2]
# #labels = torch.tensor(labels)
# labels = torch.tensor(labels)
# buffer = torch.cat((buffer, labels[labels == 2]))
# buffer = torch.cat((buffer, labels[labels == 4]))
# buffer = torch.cat((buffer, labels[labels == 6]))

# batch = []
# for element in fetal_frame.load_batch(buffer.tolist()):
#     print(element)
    
# list = [i for i in range(40)]

# del list[0:32]
# print(list)

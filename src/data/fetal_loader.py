    
import warnings
warnings.filterwarnings("ignore")
import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

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
        video = idb[2]
        ps = idb[4]
        Class = idb[1]

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
    def __init__(self, root, ann_path, transform=None, target_transform=None, img_size = 512):

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

        return images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized, 1

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
                      
    
class Video_Loader(data.Dataset):
    def __init__(self, root, videos_path, ann_path, transform=None, target_transform=None, img_scaling = False, num_frames = 4, img_size = 512):
        
        self.img_size = img_size
        self.num_frames = num_frames
        self.transform = transform
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
            print("ERROR IMG NOT LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
    
    def __len__(self):
        return len(self.videos)
    
    
    def _load_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.data.query('index == @index').iloc[0]
        
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
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize(mean=0.1354949, std=0.18222201)])
            images = t(images)
        if self.transform is not None:
            images = self.transform(images)
        return images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized
    
    
    def __getitem__(self, index):
        video = self.videos.iloc[index].values[0]
        
        batch = self.data[self.data['video'] == video] #if len(self.data[self.data['video'] == video]) > 3 else self.data[self.data['video'] == video]
        batch_len = len(batch)
        #print('batch_len', batch_len)
        starting_position = np.random.randint(0, batch_len - self.num_frames) if batch_len > self.num_frames else 0
        #print('starting_position', starting_position)
        batch_sample = batch[starting_position:starting_position+self.num_frames] if batch_len > self.num_frames else batch
        #print('batch_sample', batch_sample)
        ids = batch_sample['index']
        images = list()
        Classes = list()
        indexes = list()
        for i in ids:
            #print(i)
            image, Class, measure, ps, frame_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized = self._load_item(i)
            images.append(image)
            Classes.append(Class)
            indexes.append(index)
        images = torch.stack(images)
        Classes = torch.stack(Classes)
        assert len(images) != 0, 'images is empty'
        #print(measure)
        #print('images', images.shape)
        return images, Classes, measure, ps, frame_n, measure_scaled, indexes, days_normalized, frame_loc, measure_normalized, torch.tensor(len(images))
    


class Eval_Video_Loader(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None, img_scaling = False, num_frames = 4, img_size = 512):
        
        self.img_size = img_size
        self.num_frames = num_frames
        self.transform = transform
        self.data_path = root
        self.ann_path = ann_path
        self.data = pd.read_csv(self.ann_path)
        self.videos = pd.DataFrame(self.data['video'].unique())
        self.img_scaling = img_scaling
        
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("ERROR IMG NOT LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
    
    def __len__(self):
        return len(self.videos)
    
    
    def _load_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.data.query('index == @index').iloc[0]
        
        frame_idx = idb[0]
        Class = torch.tensor(idb[2])
        video = idb[1]
        #measure = torch.tensor(idb[3])
        ps = torch.tensor(idb[3])
        # frames_n = torch.tensor(idb[5])
        # measure_scaled = torch.tensor(idb[6], dtype=torch.float32)
        # days_normalized = torch.tensor(idb[7], dtype=torch.float32)
        # frame_loc = torch.tensor(idb[8], dtype=torch.float32)
        # height = torch.tensor(idb[9])
        # width = torch.tensor(idb[10])
        # measure_normalized = torch.tensor(idb[13], dtype=torch.float32)
        images = self._load_image(self.data_path  + frame_idx + '.png')
        # new_height = clamp(int(height*2.161191086437513), 0, 512)
        # new_width = clamp(int(width*2.161191086437513), 0, 512)
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
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize(mean=0.1354949, std=0.18222201)])
            images = t(images)
        if self.transform is not None:
            images = self.transform(images)
        return images, frame_idx, video, ps, Class
    
    
    def __getitem__(self, index):
        video = self.videos.iloc[index].values[0]
        
        batch = self.data[self.data['video'] == video] #if len(self.data[self.data['video'] == video]) > 3 else self.data[self.data['video'] == video]
        batch_len = len(batch)
        starting_position = np.random.randint(0, batch_len - self.num_frames) if batch_len > self.num_frames else 0
        batch_sample = batch[starting_position:starting_position+self.num_frames] if batch_len > self.num_frames else batch
        #print('batch_sample', batch_sample)
        ids = batch_sample['index']
        images = list()
        Classes = list()
        for i in ids:
            #print(i)
            image, frame_idx, vid, ps, Class = self._load_item(i)
            images.append(image)
            Classes.append(Class)
        images = torch.stack(images)
        Classes = torch.stack(Classes)
        assert len(images) != 0, 'images is empty'
        #print(measure)
        #print('images', images.shape)
        return images, Classes, vid, ps, torch.tensor(len(images))
    
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

# # batch = []
# # for element in fetal_frame.load_batch(buffer.tolist()):
# #     print(element)
    
# # # list = [i for i in range(40)]

# # # del list[0:32]
# # # print(list)
# # # import random
# # # np.random.seed(1)
# # # random.seed(1)
# root = '/data/kpusteln/fetal_test_data_extracted/'
# videos_path = '/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/all/videos_train.csv'
# ann_path = '/data/kpusteln/Fetal-RL/data_preparation/test_data/results_cls_combined.csv'

# def my_collate(batch):
    # images = [item[0] for item in batch]
    # Classes = [item[1] for item in batch]
    # videos = [item[2] for item in batch]
    # ps = torch.stack([item[3] for item in batch])
    # lens = [item[4] for item in batch]
    # images_lens = [len(img) for img in images]
    # assert lens == images_lens, 'images are not equal to lens'
    # return images, Classes, videos, ps, lens


# videos_data = Eval_Video_Loader(root, ann_path)


# train_loader = torch.utils.data.DataLoader(videos_data, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=my_collate)

# for i in range(10):
#     data = train_loader.__iter__().__next__()
#     images_org = data[0]
#     split_sizes = data[4]
#     images = torch.cat(images_org)
#     video = data[2]
#     ps = data[3]
#     Classes = data[1]
#     test_batch_unbinded = torch.split(images, split_size_or_sections = split_sizes, dim=0)
#     assert len(test_batch_unbinded) == len(split_sizes), 'split len is not equal to batch size*frames number'
#     for i in range(len(test_batch_unbinded)):
#         assert len(test_batch_unbinded[i]) == split_sizes[i], 'split sizes is not equal to frames number'
    
    
    
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
import torch

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if self.target_transform is not None:
            target = self.target_transform(target)
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        return images, target, score

    def __len__(self):
        return len(self.database)


t = transforms.Compose([transforms.Resize((450, 600)),
                transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
                transforms.Resize((224, 224)),
                transforms.ToTensor()])


fetal = Fetal(root = '/data/kpusteln/fetal/fetal_extracted/', ann_path = './outputs/labels_corrected.csv', transform=t)

batch_size = 1
loader = torch.utils.data.DataLoader(
      fetal,
      batch_size = batch_size,
      shuffle=False,
      num_workers = 40
  )


mean = 0.
meansq = 0.
print('Calculating mean and std...')
for i, data in enumerate(loader):
    data = data[0]
    #data = data.cuda()
    mean += data.sum()
    meansq += (data**2).sum()
    if i % 10 == 0:
        print(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')
        with open("status_mean_std_cpu.txt", "w") as text_file:
            text_file.write(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')
with open("status_mean_std_cpu.txt", "w") as text_file:
            text_file.write(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')

mean = mean/(len(fetal)*batch_size*224*224)
meansq = meansq/(len(fetal)*batch_size*224*224)
std = torch.sqrt(meansq - mean**2)
mean = mean.cpu().detach().numpy()
std = std.cpu().detach().numpy()

np.save('mean_std', [mean, std])





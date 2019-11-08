

from __future__ import  print_function,division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset, DataLoader
from torchvision import  transforms,utils
import  warnings
warnings.filterwarnings("ignore")
plt.ion()

class FacelandmarksDataset(Dataset):

    def __init__(self, csv_file, img_dir,transform=None):
        self.landmarks_frame=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.transform=transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_name=self.landmarks_frame.iloc[idx,0]
        image=io.imread(os.path.join(self.img_dir,image_name))
        landmarks=self.landmarks_frame.iloc[idx,1:]
        landmarks = np.array([landmarks])
        landmarks=landmarks.astype('float').reshape(-1,2)
        sample={'image':image,'landmarks':landmarks}
        if self.transform:
            sample=self.transform(sample)
        return  sample

class Rescale(object):
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    #convert ndarrays to tensors
    def __call__(self, sample):
        image, landmarks=sample['image'], sample['landmarks']
        #from H*W*C to C*W*H
        image=image.transpose((2,0,1))
        return {'image':torch.from_numpy(image),'landmarks':torch.from_numpy(landmarks)}

def data_transform_demo(dataset):
    scale=Rescale(256)
    crop=RandomCrop(128)
    composed=transforms.Compose([scale,crop])
    fig=plt.figure()
    n=35
    sample=dataset[n]

    for i, tsfm in enumerate([scale,crop, composed]):
        transformed_sample=tsfm(sample)
        ax=plt.subplot(1,3,i+1)
        plt.tight_layout()
        ax.set_title(type(tsfm).__name__)
        sample_data_show(**transformed_sample)
    plt.show()



def data_show(face_dataset):
    fig=plt.figure()
    for i in range(len(face_dataset)):
        sample=face_dataset[i]
        print (i, sample['image'].shape, sample['landmarks'].shape)

        ax=plt.subplot(1,4,i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        sample_data_show(**sample)
        if i==3:
            plt.show()
            break

def transformed_tensor_show(dataset):
    #fig=plt.figure()
    for i in range(len(dataset)):
        sample=dataset[i]
        print (i, sample['image'].size(), sample['landmarks'].size())

        # ax=plt.subplot(1,4,i+1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # sample_data_show(**sample)
        if i==3:
            # plt.show()
            break



def sample_data_show (image, landmarks):
    # landmarks_frame=pd.read_csv(face_csv)
    # img_name = landmarks_frame.iloc[n,0]
    # landmarks=landmarks_frame.iloc[n,1:].as_matrix()
    # landmarks=landmarks.astype('float').reshape(-1,2)
    #
    # print('Image name: {}'.format(img_name))
    # print ('landmarks shape: {}'.format(landmarks.shape))
    # print ('First 4 Landmarks: {}'.format(landmarks[:4]))
    #
    # image_path=os.path.join(data_path,img_name)
    # image=io.imread(image_path)
    # plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
    plt.pause(2)

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')



def demo_dataloader(tsfmdataset, bs, num):
    dataloader=DataLoader(dataset=tsfmdataset,batch_size=bs,shuffle=True,num_workers=num)
    for id_batch,data_batch in enumerate(dataloader):
        print("Batch Id:{}, number of images: {}, number of landmarks: {}".format(id_batch, data_batch['image'].size(), data_batch['landmarks'].size()))

        if id_batch == 3:
            plt.figure()
            show_landmarks_batch(data_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

if __name__=='__main__':

    #initialize dataset by Dataset class
    data_path='./data/faces/'
    data_csv=os.path.join(data_path, 'face_landmarks.csv')
    face_dataset=FacelandmarksDataset(data_csv,data_path)
    data_show(face_dataset)

    #initialize dataset and specify transform types
    data_transform_demo(face_dataset)
    transformed_data=FacelandmarksDataset(data_csv,data_path,transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    transformed_tensor_show(transformed_data)

    #data loader
    demo_dataloader(transformed_data, 4, 3)
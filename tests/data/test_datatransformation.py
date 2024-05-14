import os

import torch
from cell_similarity.data.datasets import ImageDataset
from dinov2.data import DataAugmentationDINO
from torch.utils.data import DataLoader

def test():

    data_transform = DataAugmentationDINO(
        (0.32, 1.0),
        (0.05, 0.32),
        8,
    )
    
    path_dataset = os.path.join(os.getcwd(), 'dataset_test')
    dataset = ImageDataset(root=path_dataset, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=32)

    for i in dataloader:
        assert len(i['global_crops']) == 2
        assert i['global_crops'][0].shape == torch.Size([32, 3, 224, 224])
        assert len(i['local_crops']) == 8
        assert i['local_crops'][0].shape == torch.Size([32, 3, 96, 96])
        break

if __name__ =='__main__':
    test()

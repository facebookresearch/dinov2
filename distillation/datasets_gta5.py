from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import torch 
from PIL import Image
import os
import numpy as np
import random
from albumentations import Compose
import math
import itertools
import torch.nn.functional as F

class GTA5(Dataset):


    """
    GTA5 Dataset class for loading and transforming GTA5 dataset images and labels for semantic segmentation tasks.

    """


    def __init__(self, GTA5_path: str, transform: Optional[Compose] = None, FDA: float = None):


        """
        Initializes the GTA5 dataset class.

        This constructor sets up the dataset for use, optionally applying Frequency Domain Adaptation (FDA) and other transformations to the data.

        Args:
            GTA5_path (str): The root directory path where the GTA5 dataset is stored.
            transform (callable, optional): A function/transform that takes in an image and label and returns a transformed version. Defaults to None.
            FDA (float, optional): The beta value for Frequency Domain Adaptation. If None, FDA is not applied. Defaults to None.
        """


        self.GTA5_path = GTA5_path
        self.transform = transform
        self.FDA = FDA
        self.data = self._load_data()
        self.color_to_id = get_color_to_id()
        self.target_images = self._load_target_images() if FDA else []

    def _load_data(self)->List[Tuple[str, str]]:

        """
        Load data paths for GTA5 dataset images and labels.

        This method walks through the directory structure of the GTA5 dataset, specifically looking for image files in the 'images' folder and corresponding label files in the 'labels' folder. It constructs a list of tuples, each containing the path to an image file and the corresponding label file.

        Returns:
            list: A list of tuples, each containing the path to an image file and the corresponding label file.
        """

        data = []
        image_dir = os.path.join(self.GTA5_path, 'images')
        label_dir = os.path.join(self.GTA5_path, 'labels')
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, image_filename)
            data.append((image_path, label_path))
        return data

    def _load_target_images(self)->List[Tuple[str, str]]:

        """
        Load target images for Frequency Domain Adaptation.

        This method walks through the directory structure of the Cityscapes dataset, specifically looking for image files in the 'gtFine' folder. It constructs a list of tuples, each containing the path to a label file and the corresponding image file.

        Returns:
            list: A list of tuples, each containing the path to a label file and the corresponding image file.
        """

        target_images = []
        city_path = self.GTA5_path.replace('GTA5', 'Cityscapes')
        city_image_dir = os.path.join(city_path, 'Cityspaces', 'gtFine', 'train')
        for root, _, files in os.walk(city_image_dir):
            for file in files:
                if 'Id' in file:
                    label_path = os.path.join(root, file)
                    image_path = label_path.replace('gtFine/', 'images/').replace('_gtFine_labelTrainIds', '_leftImg8bit')
                    target_images.append((label_path, image_path))
        return target_images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Get the image and label at the specified index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and label.
        """

        img_path, label_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        label = self._convert_rgb_to_label(Image.open(label_path).convert('RGB'))
        img, label = np.array(img), np.array(label)
        center_padding = CenterPadding(14)

        if self.FDA:
            target_image_path = random.choice(self.target_images)[1]
            target_image = Image.open(target_image_path).convert('RGB').resize(img.shape[1::-1])
            img = FDA_transform(img, np.array(target_image), beta=self.FDA)
            
        if self.transform:
            
            transformed = self.transform(image=img, mask=label)
            img, label = transformed['image'], transformed['mask']
        
        
        

        img = torch.from_numpy(img).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        return center_padding(img), center_padding(label)

    def __len__(self)->int:

        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """

        return len(self.data)
    
    def _convert_rgb_to_label(self, img:Image.Image)->np.ndarray:

        """
        Convert RGB image to grayscale label.

        Args:
            img (Image.Image): The RGB image to convert to grayscale.

        Returns:
            np.ndarray: The grayscale label image.
        """

        gray_img = Image.new('L', img.size)
        label_pixels = img.load()
        gray_pixels = gray_img.load()
        
        for i in range(img.width):
            for j in range(img.height):
                rgb = label_pixels[i, j]
                gray_pixels[i, j] = self.color_to_id.get(rgb, 255)
        
        return gray_img
    

import numpy as np
from PIL import Image
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
import PIL
def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Compute a fast histogram for evaluating segmentation metrics.

    This function calculates a 2D histogram where each entry (i, j) counts the number of pixels that have the true label i and the predicted label j with a mask.

    Args:
        a (np.ndarray): An array of true labels.
        b (np.ndarray): An array of predicted labels.
        n (int): The number of different labels.

    Returns:
        np.ndarray: A 2D histogram of size (n, n).
    """
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) for each class.

    The IoU is computed for each class using the histogram of true and predicted labels. It is defined as the ratio of the diagonal elements of the histogram to the sum of the corresponding rows and columns, adjusted by the diagonal elements and a small epsilon to avoid division by zero.

    Args:
        hist (np.ndarray): A 2D histogram where each entry (i, j) is the count of pixels with true label i and predicted label j.

    Returns:
        np.ndarray: An array containing the IoU for each class.
    """
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def poly_lr_scheduler(optimizer:torch.optim.Optimizer, init_lr:float, iter:int, lr_decay_iter:int=1,
                      max_iter:int=50, power:float=0.9)->float:
    """
    Adjusts the learning rate of the optimizer for each iteration using a polynomial decay schedule.

    This function updates the learning rate of the optimizer based on the current iteration number and a polynomial decay schedule. The learning rate is calculated using the formula:
    
        lr = init_lr * (1 - iter/max_iter) ** power
    
    where `init_lr` is the initial learning rate, `iter` is the current iteration number, `max_iter` is the maximum number of iterations, and `power` is the exponent used in the polynomial decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        init_lr (float): The initial learning rate.
        iter (int): The current iteration number.
        lr_decay_iter (int): The iteration interval after which the learning rate is decayed. Default is 1.
        max_iter (int): The maximum number of iterations after which no more decay will happen.
        power (float): The exponent used in the polynomial decay of the learning rate.

    Returns:
        float: The updated learning rate.
    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    # lr = init_lr*(1 - iter/max_iter)**power
    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def label_to_rgb(label:np.ndarray, height:int, width:int)->PIL.Image:
    """
    Transforms a label matrix into a corresponding RGB image utilizing a predefined color map.

    This function maps each label identifier in a two-dimensional array to a specific color, thereby generating an RGB image. This is particularly useful for visualizing segmentation results where each label corresponds to a different segment class.

    Parameters:
        label (np.ndarray): A two-dimensional array where each element represents a label identifier.
        height (int): The desired height of the resulting RGB image.
        width (int): The desired width of the resulting RGB image.

    Returns:
        PIL.Image: An image object representing the RGB image constructed from the label matrix.
    """
    id_to_color = get_id_to_color()
    
    height, width = label.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            class_id = label[i, j]
            rgb_image[i, j] = id_to_color.get(class_id, (255, 255, 255))  # Default to white if not found
    pil_image = Image.fromarray(rgb_image, 'RGB')
    return pil_image

def generate_cow_mask(img_size:tuple, sigma:float, p:float, batch_size:int)->np.ndarray:

    """
    Generates a batch of cow masks based on a Gaussian noise model.

    Parameters:
        img_size (tuple): The size of the images (height, width).
        sigma (float): The standard deviation of the Gaussian filter applied to the noise.
        p (float): The desired proportion of the mask that should be 'cow'.
        batch_size (int): The number of masks to generate.

    Returns:
        np.ndarray: A batch of cow masks of shape (batch_size, 1, height, width).
    """
    N = np.random.normal(size=img_size) 
    Ns = gaussian_filter(N, sigma)
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    masks = []
    for i in range(batch_size):
        masks.append((Ns > t).astype(float).reshape(1,*img_size))
    return np.array(masks)

def get_id_to_label() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding labels.

    Returns:
        dict: A dictionary where keys are class IDs and values are labels.
    """
    return {
        0: 'road',
        1: 'sidewalk',
        2: 'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'light',
        7: 'sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        255: 'unlabeled'
    }

def get_id_to_color() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding colors.

    Returns:
        dict: A dictionary where keys are class IDs and values are RGB color tuples.
    """
    id_to_color = {
        0: (128, 64, 128),    # road
        1: (244, 35, 232),    # sidewalk
        2: (70, 70, 70),      # building
        3: (102, 102, 156),   # wall
        4: (190, 153, 153),   # fence
        5: (153, 153, 153),   # pole
        6: (250, 170, 30),    # light
        7: (220, 220, 0),     # sign
        8: (107, 142, 35),    # vegetation
        9: (152, 251, 152),   # terrain
        10: (70, 130, 180),   # sky
        11: (220, 20, 60),    # person
        12: (255, 0, 0),      # rider
        13: (0, 0, 142),      # car
        14: (0, 0, 70),       # truck
        15: (0, 60, 100),     # bus
        16: (0, 80, 100),     # train
        17: (0, 0, 230),      # motorcycle
        18: (119, 11, 32),    # bicycle
    }
    return id_to_color

def get_color_to_id() -> dict:
    """
    Returns a dictionary mapping RGB color tuples to their corresponding class IDs.

    Returns:
        dict: A dictionary where keys are RGB color tuples and values are class IDs.
    """
    id_to_color = get_id_to_color()
    color_to_id = {color: id for id, color in id_to_color.items()}
    return color_to_id

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
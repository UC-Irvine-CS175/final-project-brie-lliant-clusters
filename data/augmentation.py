'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import random
from sklearn import preprocessing
import numpy as np
import torch
from typing import Any, Tuple
import torchvision
from torchvision import transforms

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        
        img_norm = preprocessing.normalize(img_array)
        return img_norm

 

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.resize_height = resize_height
        self.resize_width = resize_width
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        resize = cv2.resize(img,(self.resize_width,self.resize_height),interpolation = cv2.INTER_LINEAR)
        return resize
       

class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize

class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        Vflip = np.flipud(image)
        return Vflip
      


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        Hflip = np.fliplr(image)
        return Hflip
    


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        self.rotate = rotate

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        if self.rotate == 90:
            img_rotate = np.rot90(image,k=1,axes=(0,1))
        elif self.rotate == 180:
            img_rotate = np.rot90(image,k=2,axes=(0,1))
        elif self.rotate == 270:
            img_rotate = np.rot90(image,k=3,axes=(0,1))
        else:
            raise ValueError("Invalid rotation value. Valid values are 90, 180, or 270.")
        
        return img_rotate


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width 

    def __call__(self, image):
        height,width = image.shape[:2]
        if self.output_height > height or self.output_width > width:
            raise ValueError("Output size is larger than image size.")
        
        top = random.randint(0,height - self.output_height)
        left = random.randint(0,width - self.output_width)

        bottom = top + self.output_height
        right = left + self.output_width

        img_cropped = image[top:bottom , left:right]

        return img_cropped

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.copy(image)
        
        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor()

        # Convert the image to PyTorch tensor
        tensor = transform(image)

        return tensor

def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)

if __name__ == "__main__":
    main()
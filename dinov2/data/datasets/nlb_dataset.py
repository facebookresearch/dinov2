import pathlib
from typing import Optional, Callable

from PIL import Image

from dinov2.data.datasets.extended import ExtendedVisionDataset


class NLBDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.root = pathlib.Path(root)
        self.images_paths = list(self.root.iterdir())


    def load_image(self, index: int):
        """Opens an image via a path and returns it."""
        image_path = self.images_paths[index]
        return Image.open(image_path).convert("RGB")

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array
        
        img = self.load_image(index)
        img = img.tobytes()

        return img

    def get_target(self, index: int):
        image_path = self.images_paths[index]
        return image_path

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images_paths)
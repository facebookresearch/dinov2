import os
import csv
import requests
import json
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from PIL import Image
from torch.utils.data import Dataset

import os
import csv
import requests
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

class RaiseDataset(Dataset):
    def __init__(
        self,
        root: str,
        download_dir: str = "./raw_nod_images",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        RAISE Dataset for image classification.

        Args:
            root (str): Path to the dataset CSV file.
            download_dir (str): Directory where images will be stored.
            transform (Callable, optional): Image transformations.
            target_transform (Callable, optional): Transformations for labels.
        """
        self.download_dir = Path(download_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_info = []
        
        self._load_csv(root)

    def _load_csv(self, csv_file: str) -> None:
        """Loads dataset from the CSV file and downloads images if missing."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            i = 0
            for row in reader:
                if i > 100:
                    break
                nef_url = row["NEF"]
                file_name = os.path.basename(nef_url)
                file_path = self.download_dir / file_name

                # Get labels (last and second-to-last columns)
                labels = [row["Keywords"], row["Scene Mode"]]

                self.image_info.append({
                    "nef_url": nef_url,
                    "file_path": file_path,
                    "labels": labels
                })
                
                # Download image if not already present
                if not file_path.exists():
                    self._download_image(nef_url, file_path)
                i += 1

        print(f"Dataset loaded: {len(self.image_info)} images.")

    def _download_image(self, url: str, file_path: Path) -> None:
        """Downloads an image from a URL."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {file_path.name}")
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[str]]:
        """Returns image and corresponding labels."""
        info = self.image_info[idx]
        image = Image.open(info["file_path"]).convert("RGB")
        labels = info["labels"]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels

    def get_targets(self) -> List[List[str]]:
        """Returns all labels in dataset."""
        return [info["labels"] for info in self.image_info]

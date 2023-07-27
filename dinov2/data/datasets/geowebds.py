import torch
from torch.utils.data import IterableDataset
from osgeo import gdal
import numpy as np
import webdataset as wds
from typing import Any, Callable, List, Optional, Set, Tuple


typemapping_gdal_to_numpy = {
  1: "uint8",
  2: "uint16",
  3: "int16",
  4: "uint32",
  5: "int32",
  6: "float32",
  7: "float64",
  10: "complex64",
  11: "complex128",
}


class GeoWebDS(IterableDataset):
    def __init__(
            self,
            *,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        # self.batchsize = 32
        self.cropsize = 320
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        num_shards = 32
        imgs_per_shard = 256
        num_nodes = 1
        num_workers = 4
        # self.num_patches = num_shards * imgs_per_shard * (2240 // self.cropsize)**2
        self.num_patches = 1000000000000  # set it to sth really high for now, so that the generator doesnt get exhausted during trainng
        self.dataset = wds.DataPipeline(
                                        # wds.SimpleShardList(root),
                                        wds.ResampledShards(root),
                                        wds.shuffle(8),
                                        wds.split_by_node,
                                        wds.split_by_worker,
                                        wds.tarfile_to_samples(),
                                        wds.to_tuple("tif"),
                                        wds.map(GeoWebDS.preprocess),
                                        self.slicer,
                                        wds.shuffle(256),
                                        wds.map(self.transform),    
                                        wds.map(GeoWebDS.fake_target)
                                    ).with_length(self.num_patches)

    @staticmethod
    def read_geotif_from_bytestream(data: bytes) -> np.ndarray:
        gdal.FileFromMemBuffer("/vsimem/tmp", data)
        ds = gdal.Open("/vsimem/tmp")
        bands = ds.RasterCount
        ys = ds.RasterYSize
        xs = ds.RasterXSize
        # dtype = typemapping_gdal_to_numpy[ds.GetRasterBand(1).DataType]
        arr = np.empty((bands, ys, xs), dtype="float32")  # CHW
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            arr[b - 1, :, :] = band.ReadAsArray()
        return torch.from_numpy(arr) / 255

    @staticmethod
    def preprocess(sample):
        return GeoWebDS.read_geotif_from_bytestream(sample[0])

    @staticmethod
    def slice_image(samples, tilesize: int):
        for img in samples:
            for y in range(0, img.shape[1], tilesize):
                for x in range(0, img.shape[2], tilesize):
                    yield img[:, y:y + tilesize, x:x + tilesize]  # CHW

    def slicer(self, img):
        return GeoWebDS.slice_image(img, self.cropsize)

    @staticmethod
    def fake_target(x):
        return x, 0

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.num_patches
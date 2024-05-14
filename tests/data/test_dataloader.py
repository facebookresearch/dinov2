import os

import torch
from functools import partial
from pathlib import Path
from cell_similarity.data.datasets import ImageDataset
from cell_similarity.data.collate import collate_data_and_cast
from dinov2.data import DataAugmentationDINO, MaskingGenerator, SamplerType, make_data_loader
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup
from dinov2.train.train import get_args_parser

def test_single_path(cfg):
    
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size
    )
    inputs_dtype = torch.half

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    
    path_dataset_test = os.path.join(os.getcwd(), 'dataset_test')
    dataset = ImageDataset(root=path_dataset_test, transform=data_transform)

    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        sampler_type=sampler_type,
        sampler_advance=0,  
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    for i in data_loader:
        assert i['collated_global_crops'].shape[0] == cfg.train.batch_size_per_gpu * 2 
        assert i['collated_local_crops'].shape[0] == cfg.train.batch_size_per_gpu * cfg.crops.local_crops_number
        break

def test_several_paths(cfg):
    
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size
    )
    inputs_dtype = torch.half

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    base_path = Path(os.getcwd())
    dirs = ['dataset_test', 'dataset_bis']
    dataset = ImageDataset(root=dirs, transform=data_transform)
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        sampler_type=sampler_type,
        sampler_advance=0,  
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    for i in data_loader:
        assert i['collated_global_crops'].shape[0] == cfg.train.batch_size_per_gpu * 2 
        assert i['collated_local_crops'].shape[0] == cfg.train.batch_size_per_gpu * cfg.crops.local_crops_number
        break

if __name__ == '__main__':
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)
    test_single_path(cfg)
    print("test_single_path succesfull")
    test_several_paths(cfg)
    print("test_several_paths successfull")

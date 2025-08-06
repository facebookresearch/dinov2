import hydra
import lightning as pl
import optimalssl as ossl
import torch
import torchmetrics

from transformers import AutoImageProcessor, AutoModelForImageClassification
from omegaconf import DictConfig
from optimalssl.data import transforms
from pathlib import Path
from loguru import logger as logging


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def run_probing(cfg: DictConfig):
    processor = AutoImageProcessor.from_pretrained(cfg.model.model_name)

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(cfg.crops.global_crops_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(mean=processor.image_mean, std=processor.image_std),
    )

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize(256),
        transforms.CenterCrop(cfg.crops.global_crops_size),
        transforms.ToImage(mean=processor.image_mean, std=processor.image_std),
    )

    col_rename = {}

    if cfg.data.input_column_name != "image":
        col_rename[cfg.data.input_column_name] = "image"
    if cfg.data.label_column_name != "label":
        col_rename[cfg.data.label_column_name] = "label"

    train = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path=cfg.data.dataset_name,
            name=cfg.data.get("config_name", None),
            split=cfg.data.train_split,
            transform=train_transform,
            rename_columns=col_rename,
        ),
        batch_size=cfg.optimization.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    val_dataset = ossl.data.HFDataset(
        path=cfg.data.dataset_name,
        name=cfg.data.get("config_name", None),
        split=cfg.data.val_split,
        transform=val_transform,
        rename_columns=col_rename,
    )

    val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.optimization.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    data = ossl.data.DataModule(train=train, val=val)

    backbone = AutoModelForImageClassification.from_pretrained(cfg.model.model_name)

    # create model

    num_classes = val_dataset.dataset.features[cfg.data.label_column_name].num_classes

    if isinstance(backbone.classifier, torch.nn.Linear):
        embedding_dim = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
    else:
        embedding_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = torch.nn.Identity()

    def forward(self, batch, stage):
        with torch.inference_mode():
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            # TODO add pooling support
            batch["norm"] = batch["embedding"].norm(dim=-1)

        return batch

    module = ossl.Module(
        backbone=ossl.backbone.EvalOnly(backbone), forward=forward, optim=None
    )

    # wandb_logger = WandbLogger(
    #     project="my-awesome-project", name="experiment-name", log_model=True
    # )

    callbacks = []
    log_keys = [
        "embedding",
        "norm",
    ]

    # create linear probe
    if cfg.eval.linear_probe:
        callbacks.append(
            ossl.callbacks.OnlineProbe(
                "linear_probe",
                module,
                "embedding",
                "label",
                probe=torch.nn.Linear(embedding_dim, num_classes),
                loss_fn=torch.nn.CrossEntropyLoss(),
                metrics=torchmetrics.classification.MulticlassAccuracy(num_classes),
            )
        )

        log_keys.append("linear_probe_preds")

    # create knn probe
    if cfg.eval.knn_probe:
        callbacks.append(
            ossl.callbacks.OnlineKNN(
                module,
                "knn_probe",
                "embedding",
                "label",
                metrics=torchmetrics.classification.MulticlassAccuracy(num_classes),
                features_dim=embedding_dim,
                **cfg.knn_probe_kwargs,
            )
        )

        log_keys.append("knn_probe_preds")

    # create writer
    save_path = (
        Path(cfg.output_dir)
        / cfg.data.dataset_name.replace("/", "_")
        / cfg.model.model_name.replace("/", "_")
    )

    writer = ossl.callbacks.OnlineWriter(
        names=log_keys,
        path=save_path,
        during=["validation"],
        every_k_epochs=0,
        save_last_epoch=True,
    )

    callbacks.append(writer)

    trainer = pl.Trainer(
        max_epochs=cfg.optimization.epochs,
        num_sanity_val_steps=10,
        callbacks=callbacks,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )

    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()
    # manager.validate()


if __name__ == "__main__":
    run_probing()

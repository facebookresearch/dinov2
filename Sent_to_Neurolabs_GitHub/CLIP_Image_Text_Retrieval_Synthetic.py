import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers import THEOPOULA, MOMENTUM_THEOPOULA
import open_clip
from tqdm import tqdm
import re
import random
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="CLIP Retrieval Training with Optimizer Comparison")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1e+14)
    parser.add_argument("--r", type=float, default=0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizers", type=str, nargs='+', default=['THEOPOULA', 'MOMENTUM_THEOPOULA', 'sgd', 'adamw'])
    parser.add_argument("--data_root", type=str, default='/home/sbruno/Documents/Neurolabs-dinov2/dataset_syn_abi/dataset_syn')
    parser.add_argument("--train_csv", type=str, default='cleaned_train_all_descriptions.csv')
    parser.add_argument("--val_csv", type=str, default='filtered_cleaned_val_all_descriptions.csv')
    parser.add_argument("--k", type=int, default=5, help="Recall@k")
    parser.add_argument("--output_csv", type=str, default="optimizer_comparison_summary.csv")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path, delimiter='\t', quotechar='"', engine='python')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for _ in range(len(self.data)):
            row = self.data.iloc[idx]
            img_path = os.path.join(self.img_dir, row['image_path'].strip())
            if os.path.isfile(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                text = row['description']
                return image, text
            else:
                print(f"Warning: Missing image at {img_path}, skipping index {idx}.")
                idx = (idx + 1) % len(self.data)
        raise RuntimeError("No valid images found in dataset!")


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def get_data_loaders(train_csv, val_csv, img_dir, transform, batch_size, num_workers):
    train_dataset = CLIPDataset(train_csv, img_dir, transform)
    val_dataset = CLIPDataset(val_csv, img_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_model_and_tokenizer(device):
    model_name = 'ViT-B-32-quickgelu'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e32')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


try:
    clip_loss_fn = open_clip.loss.ClipLoss()
except AttributeError:
    from open_clip.loss import clip_loss as clip_loss_fn

import torch.nn.functional as F
def manual_clip_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def train_clip(model, loader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0
    count = 0
    for images, texts in tqdm(loader, desc="Training"):
        images = images.to(device)
        texts = tokenizer(list(texts)).to(device)
        optimizer.zero_grad()
        output = model(images, texts)
        if isinstance(output, tuple) and len(output) >= 2:
            logits_per_image, logits_per_text = output[:2]
        else:
            raise RuntimeError("Model did not return (logits_per_image, logits_per_text, ...)")
        try:
            loss = clip_loss_fn(logits_per_image, logits_per_text, model.logit_scale.exp())
        except Exception:
            loss = manual_clip_loss(logits_per_image, logits_per_text)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    return total_loss / count


def normalize_description(desc):
    desc = desc.lower().strip()
    desc = re.sub(r"[^a-z0-9 ]+", "", desc)
    return desc

def build_group_map(csv_path, delimiter='\t', desc_col='description'):
    df = pd.read_csv(csv_path, delimiter=delimiter, quotechar='"', engine='python')
    group_keys = df[desc_col].apply(normalize_description)
    group_to_indices = {}
    for idx, key in enumerate(group_keys):
        group_to_indices.setdefault(key, []).append(idx)
    return group_keys.tolist(), group_to_indices

def eval_clip_grouped(model, loader, tokenizer, csv_path, device, k=5):
    model.eval()
    image_embeds = []
    text_embeds = []
    all_texts = []
    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Encoding"):
            images = images.to(device)
            texts_tok = tokenizer(list(texts)).to(device)
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(texts_tok)
            image_embeds.append(img_emb.cpu())
            text_embeds.append(txt_emb.cpu())
            all_texts.extend(list(texts))
    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)
    sim = image_embeds @ text_embeds.t()

    group_keys, group_to_indices = build_group_map(csv_path)
    sample_groups = [normalize_description(t) for t in all_texts]

    recall_at_k = []
    for i in range(sim.size(0)):
        sims = sim[i]
        topk = sims.topk(k)[1].tolist()
        group = sample_groups[i]
        correct_indices = set(group_to_indices[group])
        is_correct = any(idx in correct_indices for idx in topk)
        recall_at_k.append(float(is_correct))
    return sum(recall_at_k) / len(recall_at_k)


def get_optimizer(optim_name, params, lr, eta, beta, r, eps, weight_decay, momentum):
    if optim_name == 'THEOPOULA':
        optimizer = THEOPOULA(params, lr=lr, eta=eta, beta=beta, r=r, eps=eps, weight_decay=weight_decay)
    elif optim_name == 'MOMENTUM_THEOPOULA':
        optimizer = MOMENTUM_THEOPOULA(params, lr=lr, eta=eta, beta=beta, r=r, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'sgd':
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == 'adamw':
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
    return optimizer


def build_summary_table(results):
    rows = []
    for optim_name, history in results.items():
        for epoch, (loss, recall) in enumerate(zip(history['train_loss'], history['recall5_grouped']), 1):
            rows.append({
                "optimizer": optim_name,
                "epoch": epoch,
                "train_loss": loss,
                "recall5_grouped": recall,
            })
    df = pd.DataFrame(rows)
    return df


def run_experiment(
    optim_name,
    device,
    batch_size,
    epochs,
    lr,
    eta,
    beta,
    r,
    eps,
    weight_decay,
    momentum,
    num_workers,
    img_size,
    seed,
    train_csv,
    val_csv,
    img_dir,
    k
):
    print(f"\n****** Choice optimizer: {optim_name} ******")
    set_seed(seed)  
    transform = get_transform(img_size)
    train_loader, val_loader = get_data_loaders(train_csv, val_csv, img_dir, transform, batch_size, num_workers)
    model, tokenizer = get_model_and_tokenizer(device)
    optimizer = get_optimizer(
        optim_name,
        model.parameters(),
        lr=lr,
        eta=eta,
        beta=beta,
        r=r,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum
    )
    history = {'train_loss': [], 'recall5_grouped': []}
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_clip(model, train_loader, optimizer, tokenizer, device)
        recall5_grouped = eval_clip_grouped(model, val_loader, tokenizer, val_csv, device, k=k)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Grouped Recall@{k}: {recall5_grouped:.4f}")
        history['train_loss'].append(train_loss)
        history['recall5_grouped'].append(recall5_grouped)
    return history

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_ROOT = args.data_root
    IMG_DIR = os.path.join(DATA_ROOT, 'img')
    TRAIN_CSV = os.path.join(DATA_ROOT, args.train_csv)
    VAL_CSV = os.path.join(DATA_ROOT, args.val_csv)

    results = {}
    for optim_name in args.optimizers:
        results[optim_name] = run_experiment(
            optim_name=optim_name,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            eta=args.eta,
            beta=args.beta,
            r=args.r,
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            num_workers=args.num_workers,
            img_size=args.img_size,
            seed=args.seed,
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            img_dir=IMG_DIR,
            k=args.k
        )
    
    print("\n*** Summary Table ***")
    for optim_name, history in results.items():
        print(f"\nOptimizer: {optim_name}")
        for epoch, (loss, recall) in enumerate(zip(history['train_loss'], history['recall5_grouped']), 1):
            print(f"Epoch {epoch:2d} | Train Loss: {loss:.4f} | Recall@{args.k}: {recall:.4f}")

    
    summary_df = build_summary_table(results)
    summary_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for opt in summary_df['optimizer'].unique():
            df_sub = summary_df[summary_df['optimizer'] == opt]
            plt.plot(df_sub['epoch'], df_sub['recall5_grouped'], label=opt)
        plt.title(f"Grouped Recall@{args.k} vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(f"Grouped Recall@{args.k}")
        plt.legend()
        plt.subplot(1, 2, 2)
        for opt in summary_df['optimizer'].unique():
            df_sub = summary_df[summary_df['optimizer'] == opt]
            plt.plot(df_sub['epoch'], df_sub['train_loss'], label=opt)
        plt.title("Train Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print("No plotting.")

    
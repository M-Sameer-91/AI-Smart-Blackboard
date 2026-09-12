"""Reproducible trainer for the existing 62-class handwriting CNN."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from app.recognition.handwriting.dataset import HandwritingDataset
from app.recognition.handwriting.model import HandwritingCNN
from app.recognition.handwriting.preprocessing import HandwritingPreprocessor

def seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def split(labels: list[str], ratio: float, value: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels): groups.setdefault(label, []).append(index)
    train, validation, rng = [], [], random.Random(value)
    for entries in groups.values():
        rng.shuffle(entries); count = max(1, round(len(entries) * ratio)); validation += entries[:count]; train += entries[count:]
    return train, validation

def epoch(model, loader, criterion, optimizer, device):
    model.train(optimizer is not None); loss_total = correct = count = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        logits = model(images); loss = criterion(logits, labels)
        if optimizer:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
        loss_total += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); count += labels.size(0)
    return loss_total / max(count, 1), 100 * correct / max(count, 1)

def train(data: Path, output: Path, epochs: int = 35, batch_size: int = 96, validation_ratio: float = .15, seed_value: int = 42) -> dict:
    seed(seed_value); output.mkdir(parents=True, exist_ok=True); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prep = HandwritingPreprocessor(target_size=64); dataset = HandwritingDataset(str(data), transform=lambda image: prep.preprocess(image, True))
    if dataset.num_classes != 62: raise ValueError(f'Expected 62 classes, found {dataset.num_classes}')
    train_ids, validation_ids = split([item['label'] for item in dataset.data], validation_ratio, seed_value)
    loader_args = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, **loader_args); validation_loader = DataLoader(Subset(dataset, validation_ids), shuffle=False, **loader_args)
    model = HandwritingCNN(62, input_size=64).to(device); criterion = nn.CrossEntropyLoss(); optimizer = AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4); scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=.5)
    history, best, checkpoint = [], -1., output / 'checkpoint_best.pth'
    for number in range(1, epochs + 1):
        train_loss, train_accuracy = epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad(): validation_loss, validation_accuracy = epoch(model, validation_loader, criterion, None, device)
        scheduler.step(validation_accuracy); entry = {'epoch': number, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': validation_loss, 'val_accuracy': validation_accuracy}; history.append(entry); print(f"epoch {number:02d}: train {train_accuracy:.2f}% / validation {validation_accuracy:.2f}%")
        if validation_accuracy > best:
            best = validation_accuracy
            torch.save({'epoch': number, 'best_val_accuracy': best, 'num_classes': 62, 'target_size': 64, 'label_to_idx': dataset.label_to_idx, 'idx_to_label': dataset.idx_to_label, 'model_state_dict': model.state_dict()}, checkpoint)
    (output / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (output / 'class_mappings.json').write_text(json.dumps({'label_to_idx': dataset.label_to_idx, 'idx_to_label': dataset.idx_to_label, 'num_classes': 62}, indent=2), encoding='utf-8')
    return {'checkpoint': str(checkpoint), 'best_validation_accuracy': best, 'samples': len(dataset), 'classes': 62}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--data', default='app/data/datasets/handwriting/manifest.csv'); parser.add_argument('--output', default='app/data/models/handwriting'); parser.add_argument('--epochs', type=int, default=35); parser.add_argument('--batch-size', type=int, default=96); args = parser.parse_args(); print(train(Path(args.data), Path(args.output), args.epochs, args.batch_size))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch_optimizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Конфигурация
BATCH_SIZE = 32
NUM_EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data/archive (2)/images/Images'
PLOTS_DIR = './plots'

# Создаем папку для графиков, если нет
os.makedirs(PLOTS_DIR, exist_ok=True)


def prepare_data():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Папка {DATA_DIR} не найдена!")

    full_dataset = datasets.ImageFolder(root=DATA_DIR)

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    train_idx, val_idx, test_idx = random_split(
        indices, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, transform=train_transform), train_idx.indices)
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, transform=val_transform), val_idx.indices)
    test_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, transform=val_transform), test_idx.indices)

    loaders = {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    }

    return loaders, full_dataset.classes


def get_model(pretrained=True):
    weights = models.VGG11_Weights.DEFAULT if pretrained else None
    model = models.vgg11(weights=weights)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 120)
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return loss, precision, recall, f1, all_labels, all_preds


def save_training_curves(history, title):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # График Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1 Score', color='orange')
    plt.title(f'{title} - Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("+", "plus")
    filename = os.path.join(PLOTS_DIR, f"{safe_title}_curves.png")

    plt.savefig(filename)
    plt.close()
    print(f"График обучения сохранен в: {filename}")


def save_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    safe_title = title.replace(" ", "_").replace("+", "plus")
    filename = os.path.join(PLOTS_DIR, f"{safe_title}_conf_matrix.png")

    plt.savefig(filename) 
    plt.close()
    print(f"Матрица ошибок сохранена в: {filename}")

def run_experiment(exp_name, pretrained, optimizer_name, loaders, classes):
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {exp_name}")
    print(f"{'=' * 60}")

    model = get_model(pretrained=pretrained)
    criterion = nn.CrossEntropyLoss()

    lr = 1e-4 if pretrained else 1e-3

    if optimizer_name == 'Lars':
        optimizer = torch_optimizer.LARS(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
            trust_coefficient=0.001,
            eps=1e-8
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError("Unknown Optimizer")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(NUM_EPOCHS):
        t_loss, t_acc = train_epoch(model, loaders['train'], criterion, optimizer)
        v_loss, v_p, v_r, v_f1, _, _ = evaluate(model, loaders['val'], criterion)

        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(v_f1)

        print(f"Epoch {epoch + 1:02d} | TLoss: {t_loss:.4f} | VLoss: {v_loss:.4f} | F1: {v_f1:.4f}")

    _, test_p, test_r, test_f1, y_true, y_pred = evaluate(model, loaders['test'], criterion)
    print(f"--> TEST RESULTS: Precision={test_p:.4f}, Recall={test_r:.4f}, F1={test_f1:.4f}")

    save_training_curves(history, exp_name)
    save_confusion_matrix(y_true, y_pred, exp_name)

    return test_p, test_r, test_f1


def main():
    loaders, classes = prepare_data()
    results = {}

    scenarios = [
        ("Pretrained + LARS", True, "Lars"),
        ("Pretrained + Adam", True, "Adam"),
        ("Scratch + LARS", False, "Lars"),
        ("Scratch + Adam", False, "Adam")
    ]

    for name, is_pretrained, opt_name in scenarios:
        p, r, f1 = run_experiment(name, is_pretrained, opt_name, loaders, classes)
        results[name] = (p, r, f1)

    print("\n" + "=" * 65)
    print(f"{'Experiment':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for key, (p, r, f) in results.items():
        print(f"{key:<25} | {p:.4f}     | {r:.4f}     | {f:.4f}")
    print("=" * 65)
    print(f"Все графики сохранены в папку: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import DDANet
from metrics import DiceBCELoss


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# DiceLoss, CombinedLoss,
def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    return images, masks

def load_data(path):
    names_path = f"{path}/train_val.txt"
    images, masks = load_names(path, names_path)
    return images, masks

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_accuracy = 0.0
    epoch_specificity = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []
        batch_accuracy = []
        batch_specificity = []

        y_pred = torch.sigmoid(y_pred)
        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])
            batch_accuracy.append(score[4])
            batch_specificity.append(score[5])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)
        epoch_accuracy += np.mean(batch_accuracy)
        epoch_specificity += np.mean(batch_specificity)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)
    epoch_accuracy = epoch_accuracy / len(loader)
    epoch_specificity = epoch_specificity / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_accuracy, epoch_specificity]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_accuracy = 0.0
    epoch_specificity = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []
            batch_accuracy = []
            batch_specificity = []

            y_pred = torch.sigmoid(y_pred)
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                batch_accuracy.append(score[4])
                batch_specificity.append(score[5])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            epoch_accuracy += np.mean(batch_accuracy)
            epoch_specificity += np.mean(batch_specificity)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)
        epoch_accuracy = epoch_accuracy / len(loader)
        epoch_specificity = epoch_specificity / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_accuracy, epoch_specificity]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 512
    size = (image_size, image_size)
    batch_size = 4
    num_epochs = 150
    lr = 1e-4
    early_stopping_patience = 150
    checkpoint_path = "files/checkpoint.pth"
    path = "../ICH"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    images, masks = load_data(path)
    images, masks = shuffling(images, masks)

    """ 5-fold Cross-validation """
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(images)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        train_x = [images[i] for i in train_idx]
        train_y = [masks[i] for i in train_idx]
        valid_x = [images[i] for i in valid_idx]
        valid_y = [masks[i] for i in valid_idx]

        data_str = f"Dataset Size for Fold {fold + 1}:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
        print_and_save(train_log_path, data_str)

        """ Data augmentation: Transforms """
        transform =  A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])

        """ Dataset and loader """
        train_dataset = DATASET(train_x, train_y, size, transform=transform)
        valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        """ Model """
        device = torch.device('cuda')
        model = DDANet()
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        loss_fn = DiceBCELoss(dice_weight=1, bce_weight=1)

        loss_name = "BCE Dice Loss"
        data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
        print_and_save(train_log_path, data_str)

        """ Training the model """
        best_valid_metrics = 0.0
        early_stopping_count = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
            scheduler.step(valid_loss)

            if valid_metrics[1] > best_valid_metrics:
                data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}_fold_{fold + 1}"
                print_and_save(train_log_path, data_str)

                best_valid_metrics = valid_metrics[1]
                torch.save(model.state_dict(), f"{checkpoint_path}_fold_{fold + 1}")
                early_stopping_count = 0

            elif valid_metrics[1] < best_valid_metrics:
                early_stopping_count += 1

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
            data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f} - Accuracy: {train_metrics[4]:.4f} - Specificity: {train_metrics[5]:.4f}\n"
            data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f} - Accuracy: {valid_metrics[4]:.4f} - Specificity: {valid_metrics[5]:.4f}\n"
            print_and_save(train_log_path, data_str)

            if early_stopping_count == early_stopping_patience:
                data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
                print_and_save(train_log_path, data_str)
                break
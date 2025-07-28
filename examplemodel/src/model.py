import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class CampDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.images = sorted(os.listdir(image_dir))
        self.image_dir, self.label_dir = image_dir, label_dir
        self.transform, self.target_transform = transform, target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        lbl_path = os.path.join(self.label_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl


class CampDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir,
        label_dir,
        batch_size=32,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=62,
    ):
        super().__init__()
        self.image_dir, self.label_dir = image_dir, label_dir
        self.batch_size, self.val_ratio, self.test_ratio, self.seed = (
            batch_size,
            val_ratio,
            test_ratio,
            seed,
        )

    def setup(self, stage=None):
        img_tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        lbl_tf = transforms.Compose(
            [
                transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        full = CampDataset(self.image_dir, self.label_dir, img_tf, lbl_tf)
        n = len(full)
        t = int(n * self.test_ratio)
        v = int(n * self.val_ratio)
        train, val, test = random_split(
            full, [n - v - t, v, t], generator=torch.Generator().manual_seed(self.seed)
        )
        self.train_ds, self.val_ds, self.test_ds = train, val, test

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


class RefugeeCampDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        c1 = F.relu(self.conv1(x))
        p1 = self.pool1(c1)

        c2 = F.relu(self.conv2(p1))
        p2 = self.pool2(c2)

        c3 = F.relu(self.conv3(p2))
        p3 = self.pool3(c3)

        # Bottleneck
        c4 = F.relu(self.conv4(p3))

        # Decoder path
        u1 = self.up1(c4)
        u2 = self.up2(u1)
        u3 = self.up3(u2)

        return torch.sigmoid(self.outc(u3))


def pixel_acc(pred, target, thr=0.5):
    b = (pred > thr).float()
    return (b == target).float().mean()


class LitRefugeeCamp(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = RefugeeCampDetector()
        self.criterion = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = pixel_acc(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

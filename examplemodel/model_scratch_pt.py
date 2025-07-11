import os

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def get_input_example(dataloader, device):
    try:
        sample_images, _ = next(iter(dataloader))
        return sample_images[:1].to(device)
    except StopIteration:
        return torch.zeros(1, 3, 256, 256).to(device)


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    correct = (pred_binary == target).sum().item()
    total = target.numel()  
    return correct / total

class CampDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform # this is for image 
        self.target_transform = target_transform # this is for label hai guys
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L') # grayscale 0-255

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def create_data_loaders(image_dir, label_dir, batch_size=32, 
                        val_ratio=0.15, test_ratio=0.15, seed=42):

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # i took it from imagenet dataset
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    

    full_dataset = CampDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=image_transform,
        target_transform=target_transform
    )
    

    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size - test_size
    
    print(f"Total dataset size: {dataset_size}")
    print(f"Train set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, full_dataset


class RefugeeCampDetector(nn.Module):
    def __init__(self):
        super(RefugeeCampDetector, self).__init__()
        # encodder - downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256→128
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128→64
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64→32
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # decoder - upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32→64
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64→128
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 128→256
        
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))
        pool1 = self.pool1(conv1)
        
        conv2 = F.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        
        conv3 = F.relu(self.conv3(pool2))
        pool3 = self.pool3(conv3)
        
        conv4 = F.relu(self.conv4(pool3))
        
        # Decoder
        up1 = F.relu(self.upconv1(conv4))
        up2 = F.relu(self.upconv2(up1))
        up3 = F.relu(self.upconv3(up2))
        
        out = torch.sigmoid(self.output_conv(up3))
        
        return out


def train(image_dir, label_dir, num_epochs=10, batch_size=32):
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment("refugee-camp-detection")
    with mlflow.start_run():
        
        lr = 0.001
        mlflow.log_params(
                params={
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "image_dir": image_dir,
                    "label_dir": label_dir,
                    "dataset_size": len(os.listdir(image_dir)),
                }
            )


        train_loader, val_loader, test_loader, dataset = create_data_loaders(
            image_dir=image_dir,
            label_dir=label_dir,
            batch_size=batch_size
        )
        mlflow.log_input(dataset, context="training")
        # mlflow.log_artifact(image_dir, artifact_path="images")
        # mlflow.log_artifact(label_dir, artifact_path="labels")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RefugeeCampDetector().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_accuracy = 0.0
        best_model_path = "best_model.pth"
        print("Using device:", device)
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0.0
            train_accuracy = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device) 
                # forward pass
                optimizer.zero_grad()
                outputs = model(images)
                batch_loss = criterion(outputs, labels)

                batch_train_accuracy = calculate_pixel_accuracy(outputs, labels)
                train_accuracy += batch_train_accuracy

                # backward pass and optimize
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()

            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
                    batch_val_accuracy = calculate_pixel_accuracy(outputs, labels)
                    val_accuracy += batch_val_accuracy
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            
                    batch_test_accuracy = calculate_pixel_accuracy(outputs, labels)
                    test_accuracy += batch_test_accuracy

            avg_test_loss = test_loss / len(test_loader)
            avg_test_accuracy = test_accuracy / len(test_loader)

                    
            avg_train_loss = train_loss / len(train_loader)
            avg_train_accuracy = train_accuracy / len(train_loader)
    

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            

            print(f'Epoch - {epoch+1}/{num_epochs}:')
            print(f'Train -  Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')
            print(f'Val -  Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}')
            print(f'Test - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}')
                

            # model_info = mlflow.pytorch.log_model(
            #     pytorch_model=model,
            #     name=f"model-epoch-{epoch}",
            #     step=epoch,
            #     input_example=sample_input_to_log,
            # )

            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {epoch+1} with accuracy: {best_val_accuracy:.4f}")



            mlflow.log_metrics(
                metrics={
                    "train_loss": avg_train_loss,
                    "train_accuracy": avg_train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": avg_val_accuracy,
                    "test_loss": avg_test_loss,
                    "test_accuracy": avg_test_accuracy
                },
                step=epoch
            )


        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            sample_input_to_log = get_input_example(train_loader, device).cpu().detach().numpy()

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model", 
                input_example=sample_input_to_log,
                registered_model_name="best-refugeecamp-detector"
            )
            print("Successfully logged the best model to MLflow.")

            # os.remove(best_model_path)

        print("Training complete.")
        torch.save(model.state_dict(), 'checkpoint.pth')
        print("Model saved to checkpoint.pth")


if __name__ == "__main__":

    TRAINING_DEST_DIR = os.path.join(os.getcwd(), "data/train/banepa")

    image_dir = os.path.join(TRAINING_DEST_DIR, "chips")
    label_dir = os.path.join(TRAINING_DEST_DIR, "labels")

    train(image_dir, label_dir, num_epochs=15)
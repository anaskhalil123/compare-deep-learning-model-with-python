import time

import cv2
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# video_prepare
class VideoDataset(Dataset):
    def __init__(self, root_dir, classes, num_frames=16):
        self.root_dir = root_dir
        self.classes = classes
        self.num_frames = num_frames
        self.samples = []

        for label, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            for video in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, video), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]  # get from the list
        frames = self.load_video(video_path)
        return frames, label

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // self.num_frames, 1)

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        frames = np.array(frames)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        return frames


# CNN3D
class CNN3D(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN3D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.fc = nn.Linear(32 * 4 * 28 * 28, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# CNN_LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(32 * 28 * 28, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        cnn_out = []
        for t in range(seq_len):
            out = self.cnn(x[:, t])
            out = out.reshape(batch_size, -1)
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)

        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1])

        return out


# Two-Stream
class TwoStream(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(TwoStream, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)

        combined = torch.cat((out1, out2), dim=1)
        final = self.fc(combined)
        return final


device = torch.device("cpu")

model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# create train loader
classes = ["Archery", "BabyCrawling", "Basketball", "Biking"]

train_dataset = VideoDataset(
    root_dir=r"D:\University\Master Now\الفصل الأول عام 2025\multimedia systems\assignments\final assignment (Research)\implementation\UCF101 sample\train",
    classes=classes,
    num_frames=16,
)

val_dataset = VideoDataset(
    root_dir=r"D:\University\Master Now\الفصل الأول عام 2025\multimedia systems\assignments\final assignment (Research)\implementation\UCF101 sample\train",
    classes=classes,
    num_frames=16,
)


from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for epoch in range(10):
    model.train()
    total_loss = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Training Method
    def train_model(model, train_loader, criterion, optimizer, epochs=10):
        model.to(device)
        model.train()

        strat_time = time.time()

        for epoch in range(epochs):
            running_loss = 0.0

            for (
                videos,
                labels,
            ) in train_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
        training_time = time.time()
        return training_time

    def evaluate_model(model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for videos, labels in test_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        cm = confusion_matrix(all_labels, all_preds)
        return acc, f1, cm

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #### Now Lets compare!
    ##First 3D CNN
    model_3d = CNN3D(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3d.parameters(), lr=0.001)

    print("\nTraining 3D CNN...")
    time_3d = train_model(model_3d, train_loader, criterion, optimizer, epochs=10)

    acc_3d, f1_3d, cm_3d = evaluate_model(model_3d, val_loader)

    print("3D CNN Results:")
    print("Accuracy:", acc_3d)
    print("F1 Score:", f1_3d)
    print("Parameters:", count_parameters(model_3d))
    print("Training Time:", time_3d)
    print("Confusion Matrix:\n", cm_3d)

    # Second CNN + LSTM
    model_lstm = CNN_LSTM(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

    print("\nTraining CNN+LSTM...")
    time_lstm = train_model(model_lstm, train_loader, criterion, optimizer, epochs=10)

    acc_lstm, f1_lstm, cm_lstm = evaluate_model(model_lstm, val_loader)

    print("CNN+LSTM Results:")
    print("Accuracy:", acc_lstm)
    print("F1 Score:", f1_lstm)
    print("Parameters:", count_parameters(model_lstm))
    print("Training Time:", time_lstm)
    print("Confusion Matrix:\n", cm_lstm)

    # Third  Two-Stream
    model_twostream = TwoStream(CNN3D(4), CNN3D(4), 4)
    acc_twostream, f1_twostream, cm_twostream = evaluate_model(
        model_twostream, val_loader
    )

    print("\n===== FINAL COMPARISON =====")
    print(f"3D CNN -> Acc: {acc_3d:.4f}, F1: {f1_3d:.4f}")
    print(f"CNN+LSTM -> Acc: {acc_lstm:.4f}, F1: {f1_lstm:.4f}")
    print(f"Two-Stream -> Acc: {acc_twostream:.4f}, F1: {f1_twostream:.4f}")

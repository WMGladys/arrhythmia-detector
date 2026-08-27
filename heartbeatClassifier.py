import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

def load_and_preprocess_data(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path, header=None)
    test_df = pd.read_csv(test_csv_path, header=None)
    X_train_raw = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    X_test_raw = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    return X_train_padded, y_train, X_test_padded, y_test

class PreprocessedECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) #insert a channel dimension, resulting in shape (N, 1, 188) instead of (N, 188)
        self.y = torch.tensor(y, dtype=torch.long) #loss functions require the target labels to be of type long for classification tasks
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class 1DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(1DCNN, self).__init___()
        self.conv1 = nn.Conv1d(
            in_channels = 1,#signal with 1 input channel e.g., a single sensor line
            out_channels = 32, #32 filters to learn 32 different features
            kernel_size = 7, #each filter will look at 7 time steps at a time
            padding = 3 #padding to maintain the same length of the output as the input
        )
        self.bn1 = nn.BatchNorm1d(32) #normalize the outputof the convolutional layer, all 32 feature maps
        #nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size = 2) #look at 2 time steps at a time and take the maximum value
        self.fully_connected1 = nn.Linear(in_features = 128, out_features = 64) #fully connected layer to learn a combinaiton of features
        self.fully_connected2 = nn.Linear(in_features = 64, out_features = num_classes) #final output layer to predict a clinical outcome
        x = x.squeeze(-1) #Flatten the output of the last pooling layer to feed into the fully connected layers
        x = F.relu(self.fully_connected1(x)) #Pass through the first fully connected layer with ReLU activation
        x = self.dropout(x) #Apply dropout for regularization to prevent overfitting
        return self.fully_connected2(x) #Pass through the final fully connected layer to get the output predictions

def train_and_evaluate_model(train_csv, test_csv, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load raw data and preprocess features and labels
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_csv, test_csv)
    #Wrap Numpy arrays in PyTorch Dataset objects 
    train_dataset = PreprocessedECGDataset(X_train, y_test)
    test_dataset = PreprocessedECGDataset(X_test, y__train)
    #Create loaders to allow batch and shuffling of the PyTorch Dataset objects
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train() #Set the model to training mode: Dropout and BatchNorm updates
        running_loss = 0.0 #Initialize running loss for the epoch
        for inputs, labels in train_dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device) #Move the inputs and labels to the target device
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(train_dataset):.4f}")
    model.eval() #disable training-specific layers e.g., Dropout
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_dataset_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1) # Extract index of highest logit score
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())
    return model
def plot_model_performance_visuals(all_targets, all_predictions, target_names):
    # Set up styling
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    cm = confusion_matrix(all_targets, all_predictions)
    x = np.arange(len(target_names))
    width = 0.15  # Width of each individual bar
    colors = ["#2ca02c", "#d95f02", "#2b5c8f", "#7570b3", "#e7298a"]
    for i, pred_name in enumerate(target_names):
        offset = (i - len(target_names) / 2) * width + width / 2
        bars = ax1.bar(
            x + offset,
            cm_perc[:, i],
            width,
            label=f"Predicted: {pred_name}",
            color=colors[i],
        )
        for bar in bars:
            height = bar.get_height()
            if height > 2.0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
    ax1.set_title(
        "1. Prediction Breakdown per Actual Category (% Correct vs Mistakes)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Actual Ground-Truth Category", fontweight="bold")
    ax1.set_ylabel("Percentage of Samples (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(target_names, rotation=15)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", title="Model Prediction", fontsize=8)
    ax2 = fig.add_subplot(2, 2, 2)
    precision, recall, _, support = precision_recall_fscore_support(
        all_targets, all_predictions
    )
    x = np.arange(len(target_names))
    width = 0.35
    rects1 = ax2.bar(
        x - width / 2,
        recall * 100,
        width,
        label="Detection Rate (Recall)",
        color="#2b5c8f",
    )
    rects2 = ax2.bar(
        x + width / 2,
        precision * 100,
        width,
        label="Alert Reliability (Precision)",
        color="#d95f02",
    )
    ax2.set_title(
        "2. Detection Rate vs. Alert Reliability per Category",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Percentage (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(target_names, rotation=15)
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper right")
    ax2.bar_label(rects1, padding=3, fmt="%.0f%%", fontsize=9)
    ax2.bar_label(rects2, padding=3, fmt="%.0f%%", fontsize=9)
    ax3 = fig.add_subplot(2, 2, (3, 4))
    bars = ax3.bar(target_names, support, color="#2ca02c")
    ax3.set_title(
        "3. Test Dataset Distribution (Sample Count per Category)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_ylabel("Number of Patient Samples", fontweight="bold")
    ax3.set_yscale("log")  # Log scale to handle massive imbalance cleanly
    for bar in bars:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.15,
            f"{int(yval):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.show()
train_path = "INSERT PATH HERE"
test_path = "INSERT PATH HERE"
trained_model = train_and_evaluate_model(
    train_csv = train_path,
    test_csv = test_path,
    epochs = 5
    batch_size = 64
)

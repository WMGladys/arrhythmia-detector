import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

#1. DATA INGESTION AND PREPROCESSING
"""
Determines the shape of the raw data
Prepares raw CSV data for deep learning by reshaping it to the correct format for the Conv layer
Loads, splits, standardizes, and pads the data to ensure uniformity across all samples
performs standardization STRICTLY fitted on the training set to prevent data leakage into the test set
Prevent data leakage by fitting the scaler only on the training data and then transforming both the training and test data using the fitted scaler
"""
def load_and_preprocess_data(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path, header=None)
    test_df = pd.read_csv(test_csv_path, header=None)

    #Separate features(X) and labels(y)
    X_train_raw = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    X_test_raw = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)

    #Prevent Data Leakage: Fit the scaler only on X_train_raw and then transform both X_train_raw and X_test_raw
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw) #fit the scaler on the training data and transform on the TRAINING data
    X_test_scaled = scaler.transform(X_test_raw) #transform ONLY on TEST data

    #PADING: a single column of zeros to the end of every signal sequence, increasing the time steps from 187 to 188
    X_train_padded = np.pad(X_train_scaled, ((0, 0), (0, 1)), mode='constant')
    X_test_padded = np.pad(X_test_scaled, ((0, 0), (0, 1)), mode='constant')

    return X_train_padded, y_train, X_test_padded, y_test


#2. DATASET WRAPPER CLASS
"""
A PyTorch Dataset wrapper class to encapsulate raw data from cs files to allow custom handling
Allows DataLoader to automatically handle batching, shuffling, and parallel loading of the data
The Conv layer accepts inputs in 3 dimensions: (batch_size, channels, time_steps).
But the raw ECG data is in 2 dimensions: (time_steps, channels)
The dataset wrapper class will handle the reshaping of the data to the correct format for the Conv layer
Wrapper class inherits from torch.utils.data.Dataset
It requires the implementation of two methods: __len__ and __getitem__.
"""
class PreprocessedECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) #insert a channel dimension, resulting in shape (N, 1, 188) instead of (N, 188)
        self.y = torch.tensor(y, dtype=torch.long) #loss functions require the target labels to be of type long for classification tasks
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#3. CREATE A CONVOLUTIONAL NEURAL NETWORK
"""
Three sequential blocks with convolutional layers, batch normalization, ReLU activation, and pooling
BatchNorm, pooling, reLU activation, and convolution all happen in each block.
"""
class PerceptionLayer1DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(PerceptionLayer1DCNN, self).__init__()
        
        #A. FEATURE EXTRATION BLOCKS
        #Block 1: basic feature detection
        self.conv1 = nn.Conv1d(
            in_channels = 1,#signal with 1 input channel e.g., a single sensor line
            out_channels = 32, #32 filters to learn 32 different features
            kernel_size = 7, #each filter will look at 7 time steps at a time
            padding = 3 #padding to maintain the same length of the output as the input
        )
        self.bn1 = nn.BatchNorm1d(32) #normalize the outputof the convolutional layer, all 32 feature maps
        #nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size = 2) #look at 2 time steps at a time and take the maximum value

        #Block 2: more complex feature detection
        self.conv2 = nn.Conv1d(
                in_channels = 32, #input is the 32 feature maps from the previous blocl
                out_channels = 64, #64 filters to learn 64 different features
                kernel_size = 5, #each filter will look at 5 time steps at a time
                padding = 2 #padding to maintain the same length of the output as the input
            )
        self.bn2 = nn.BatchNorm1d(64) #normalize the output (64 feature maps) of the convolutional layer
        #nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size = 2) #Look at 2 time steps at a time and take the maximum value

        #Block 3: Summarize the features
        self.conv3 = nn.Conv1d(
            in_channels = 64, #input is the 64 feature maps from the previous block
            out_channels = 128, #128 filters to learn 128 different features
            kernel_size = 3, #each filter will look at 3 time steps at a time
            padding = 1
        )
        self.bn3 = nn.BatchNorm1d(128)
        #nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(output_size = 1) #Summarize the features by taking the average of each feature map, resulting in a single value per feature map

        #B. DEFINE FULLY CONNECTED LAYERS
        self.fully_connected1 = nn.Linear(in_features = 128, out_features = 64) #fully connected layer to learn a combinaiton of features
        self.dropout = nn.Dropout(0.3)
        self.fully_connected2 = nn.Linear(in_features = 64, out_features = num_classes) #final output layer to predict a clinical outcome

        #C. DEFINE FORWARD PASS TO SPECIFY HOW DATA FLOWS THROUGH THE NETWORK & PERFORM FLATTENING AND ACTIVATIONS
        #Define the Forward Pass to Specify How Data Flows Through the Network
    def forward(self, x):
        x = self.pool1(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )
        x = self.pool2(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )
        x = self.pool3(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )
        #Perform Flattenin
        x = x.squeeze(-1) #Flatten the output of the last pooling layer to feed into the fully connected layers
        x = F.relu(self.fully_connected1(x)) #Pass through the first fully connected layer with ReLU activation
        x = self.dropout(x) #Apply dropout for regularization to prevent overfitting
        return self.fully_connected2(x) #Pass through the final fully connected layer to get the output predictions

#4. TRAINING AND EVALUATION
"""
Has 4 distinct parts: 
1. HARDWARE SELECTION & DATA PREPARATION: Selects the either GPU or CPU for training, 
and prepares the data for training and evalution through the DataLoader class
2. IMBALANCE HANDLING: Counts occurrences of each label and calculates inverse frequencies,
preventing the model from favouring the majority class during training
3. TRAINING LOOP: Iterates through the training data, performs forward and backward passes, and updates model weights
4. EVALUATION LOOP: Evaluates the model on the test data, calculating loss and accuracy"""

def train_and_evaluate_model(train_csv, test_csv, epochs=5, batch_size=64):
    #CATEGORY 1: HARDWARE SELECTION & DATA PREPARATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load raw data and preprocess features and labels
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_csv, test_csv)
    #Wrap Numpy arrays in PyTorch Dataset objects 
    train_dataset = PreprocessedECGDataset(X_train, y_train)
    test_dataset = PreprocessedECGDataset(X_test, y_test)
    #Create loaders to allow batch and shuffling of the PyTorch Dataset objects
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #CATEGORY 2: IMBALANCE HANDLING & INITIALIZATION
    class_counts = np.bincount(y_train) #Count occurrences of each label in the training set
    class_weights = 1.0/torch.tensor(class_counts, dtype=torch.float32) #Calculate inverse frequencies to prevent the model from favouring the majority class during training
    class_weights = class_weights/class_weights.sum() #Normalize the inverse weights so that they sum to 1
    class_weights = class_weights.to(device) #Move class weights tensor to the target hardware device (GPU or CPU)
    #Initialize the model, loss function, and optimizer
    model = PerceptionLayer1DCNN(in_channels=1, num_classes=len(class_counts)).to(device) #Move the model to the target hardware device (GPU or CPU)
    criterion = nn.CrossEntropyLoss(weight=class_weights) #Use the class weights in the loss function to handle class imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Use the Adam optimizer with a learning rate of 0.001

    #CATEGORY 3: TRAINING LOOP
    for epoch in range(epochs):
        model.train() #Set the model to training mode: Dropout and BatchNorm updates
        running_loss = 0.0 #Initialize running loss for the epoch
        for inputs, labels in train_dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device) #Move the inputs and labels to the target device
            optimizer.zero_grad() #Clear previous gradients
            outputs = model(inputs) #Forward pass
            loss = criterion(outputs, labels) #Calculate loss function
            loss.backward() #Backpropagation
            optimizer.step() #Update weights
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(train_dataset):.4f}")

    #CATEGORY 4: MODEL EVALUATION
    model.eval() #disable training-specific layers e.g., Dropout
    all_predictions = []
    all_targets = []

    #Disable gradient(loss) tracking for faster inference and low mem. usage
    with torch.no_grad():
        for inputs, labels in test_dataset_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1) # Extract index of highest logit score
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())

     #Generate and display performance metrics
    """ 
    MEANING OF LABELS IN THE OUTPUT OF SCIKIT'S CLASSIFICATION_REPORT()
    1. Precision: How accurately the model predicts a case
    2. Recall: How many samples in a category the model was able to find
    3. Support: How many total samples are in a category
    4.F1-score: ensures there's no imbalance between precision and recall. A model can accurately predict a case 
    but fail to find all samples in a category, thus making its precision high and recall low. 
    F1-score ensures the model is both accurate when it speaks (precision) and thorough in what it finds (recall).
    """
    target_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    print("\n================ Classification Report ================")
    print(classification_report(all_targets, all_predictions, target_names=target_names))
    print("\n================ Confusion Matrix ================")
    print(confusion_matrix(all_targets, all_predictions))
    #Print summary that is easy for anyone to understand
    easy_understanding_output(all_targets, all_predictions, target_names)
    #Plot charts for visualized output
    plot_model_performance_visuals(all_targets, all_predictions, target_names)

    return model

#Function to Print Output in Easy-to-Understand Language
def easy_understanding_output(all_targets, all_predictions, target_names):
  from sklearn.metrics import precision_recall_fscore_support
  # Calculate precision, recall, f1, and support for each class
  precision, recall, _, support = precision_recall_fscore_support(
      all_targets, all_predictions
    )
  print("\n" + "=" * 65)
  print("        EASY-TO-READ HEALTHCARE PREDICTION SUMMARY")
  print("=" * 65)

  for i, name in enumerate(target_names):
    cases_caught = int(round(recall[i] * support[i]))
    total_cases = support[i]
    detection_rate = recall[i] * 100
    alert_accuracy = precision[i] * 100

    print(f"\nHeartbeat Category: {name.upper()}")
    print(f"  • Total Cases Tested: {total_cases:,} patient samples")
    print(
        f"  • Successfully Caught: {cases_caught:,} out of {total_cases:,} cases ({detection_rate:.1f}% detection rate)"
        )
    print(
        f"  • Prediction Reliability: {alert_accuracy:.1f}% of the times the model predicted this class, it was correct"
        )
#5. CREATE VISUALS
"""The charts display these three pieces of data:
1.Dataset Class Distribution chart: Shows the amount of data per category.
The goal is to show class imbalance
2. Confustion Matrix Heatmap or Bar graph: Shows where the model made inaccurate predictions. The bright diagonal line 
shows accurate predictions and the cells out of alignment with the line show the exact type of mistake e.g.,
mistaking a normal beat for ventricular.
3. Detection Rate (recall) vs Prediction Relaibility chart(precision): compares coverage(out of all sick patients,
did the model identify all of them?) and prediction accuracy (out of all patients identified, did the model
correctly diagnose them?)  
"""
def plot_model_performance_visuals(all_targets, all_predictions, target_names):
    # Set up styling
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))

    #VISUAL 1: Confusion Matrix Bar Graph
    ax1 = fig.add_subplot(2, 2, 1)

    cm = confusion_matrix(all_targets, all_predictions)
    # Convert absolute counts to percentages per actual class row
    cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    x = np.arange(len(target_names))
    width = 0.15  # Width of each individual bar

    # Plot a distinct bar series for each predicted class outcome
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

        # Add percentage labels above bars that are > 2% to keep chart clean
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

    # VISUAL 1: Normalized Confusion Matrix Heatmap
    # ax1 = fig.add_subplot(2, 2, 1)
    # cm = confusion_matrix(all_targets, all_predictions)
    # # Normalize by row (actual class totals) to get percentages
    # cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # sns.heatmap(
    #     cm_perc,
    #     annot=True,
    #     fmt=".1f",
    #     cmap="Blues",
    #     xticklabels=target_names,
    #     yticklabels=target_names,
    #     cbar_kws={"label": "Percentage (%)"},
    #     ax=ax1,
    # )
    # ax1.set_title(
    #     "1. Prediction Accuracy Matrix (% Correct vs Misclassified)",
    #     fontsize=12,
    #     fontweight="bold",
    # )
    # ax1.set_xlabel("Predicted Heartbeat Category", fontweight="bold")
    # ax1.set_ylabel("Actual Heartbeat Category", fontweight="bold")

    # VISUAL 2: Precision vs. Recall Comparison Bar Chart
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

    # Add values on top of bars
    ax2.bar_label(rects1, padding=3, fmt="%.0f%%", fontsize=9)
    ax2.bar_label(rects2, padding=3, fmt="%.0f%%", fontsize=9)

    # VISUAL 3: Dataset Support (Class Distribution)
    ax3 = fig.add_subplot(2, 2, (3, 4))
    bars = ax3.bar(target_names, support, color="#2ca02c")

    ax3.set_title(
        "3. Test Dataset Distribution (Sample Count per Category)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_ylabel("Number of Patient Samples", fontweight="bold")
    ax3.set_yscale("log")  # Log scale to handle massive imbalance cleanly

    # Add exact sample counts over bars
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

#6. EXECUTION: CALLING FUNCTIONS AND CLASSES
#Creat file paths to the unzipped Kaggle dataset
train_path = "./ecg_data/mitbih_train.csv"
test_path = "./ecg_data/mitbih_test.csv"
trained_model = train_and_evaluate_model(
    train_csv = train_path,
    test_csv = test_path,
    epochs = 5
    batch_size = 64
)

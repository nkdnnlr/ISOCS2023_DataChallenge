
# 
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

explore = False#True

def detect_events(df):
    # Find peaks of PID 1st derivative
    df_pid = df['pid_ug_m3'].astype('float')
    pid_1stderivative = np.gradient(df_pid.to_numpy())
    peaks_x, peaks_y = signal.find_peaks(pid_1stderivative, height=10, distance=40)
    peaks_y = peaks_y['peak_heights']
    index_events = df.iloc[peaks_x].index
    return peaks_x, index_events

def build_featuremap(df, event, k_max=24, columns=['event_sensor_01', 'event_sensor_02', 'event_sensor_03', 'event_sensor_04', 'air_H2S']):
    # columns = ['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']
    # df_interesting = df[columns].astype('float').to_numpy()
    # df_interesting = np.gradient(df_interesting, axis=1)
    subset = df[columns].astype('float').to_numpy()

    features = []
    for k in range(1, k_max+1):
        feature = subset[event+k-1: event+k]#.to_numpy()
        # feature = df_interesting.iloc[event+k-1: event+k].to_numpy()
        features.append(feature)
    features = np.array(features).flatten()
    print(features.shape)
    return features

###### Train

# Load training labels
training_labels = Path('data/Training_Dataset_ISOCSWinterSchool2023.csv')
if not training_labels.exists():
    raise ValueError(f"{training_labels} does not exist.")
labels = pd.read_csv(training_labels, sep=',')
print(labels)

# exit()
# labels = labels.drop('Sample ID')
labels['date_start'] = pd.to_datetime(labels['Date'] + ' ' + labels['Start'])
labels['date_end'] = pd.to_datetime(labels['Date'] + ' ' + labels['End'])
# print(labels.keys())


# Load training data
training_data = Path('data/WT1-30146.csv')
if not training_data.exists():
    raise ValueError(f"{training_data} does not exist.")
df = pd.read_csv(training_data, sep=';')

# Filter and preprocess data
df = df[df['element']=='WT1-30146'].dropna(subset=['air_mox0'])
df['date'] = pd.to_datetime(df['date']).apply(lambda t: t.replace(tzinfo=None))

df = df[df['date'] > pd.to_datetime('16.01.23 14:15')]
# print(df['date'].head())
# print(labels['date_start'])
# exit()

# Set labels
df['odour'] = 'ambient'
for _, row in labels.iterrows():
    # print(row)
    odour = row['Odour Type']
    for idx, df_row in df[(df['date'] >= row['date_start']) & (df['date'] <= row['date_end'])].iterrows():
        df.loc[idx, 'odour'] = row['Odour Type']
print(np.unique(df['odour'], return_counts=True))
label_colors = {
    'Downhill': 'red', 
    'Giant Slalom': 'blue', 
    'Slalom': 'green', 
    'ambient': 'gray',
}

# Set date as index
df = df.set_index('date')

if explore:
    # Explore
    # Create dataframes for each measurement type
    df_pid = df['pid_ug_m3'].astype('float')
    df_mox = df[['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']].astype('float')
    df_chem = df[['air_NH3', 'air_H2S']].astype('float')
    df_event = df[['event_sensor_01', 'event_sensor_02', 'event_sensor_03', 'event_sensor_04']].astype('float')

    # Find peaks
    # tmp_ = df_pid.resample('T').interpolate()
    # slope = pd.Series(np.gradient(tmp_.data), tmp_.index, name='slope')
    # df_pid_1stderivative = pd.concat([tmp_.rename('data'), slope], axis=1)
    pid_1stderivative = np.gradient(df_pid.to_numpy())

    peaks_x, peaks_y = signal.find_peaks(pid_1stderivative, height=10, distance=40)
    peaks_y = peaks_y['peak_heights']
    # print(pid_1stderivative.shape)
    # print(df_pid.shape)
    # exit()
    # Create the figure and axes
    fig, ax = plt.subplots(nrows=8, sharex=True)

    # Plot the data
    ax[0].scatter(df.index, [1]*len(df), c=[label_colors[l] for l in df['odour']], s=2)

    ax[1].plot(df_pid, c='r', label='PID')
    ax[1].set_ylabel('PID')
    # ax[1].scatter(df_pid.iloc[peaks_x] c='k')
    df_pid.iloc[peaks_x].plot(style='.', ax=ax[1], c='k')

    for i, chem_column in enumerate(df_chem.columns):
        ax[i+2].plot(df_chem[chem_column], c='b', label=chem_column)
        ax[i+2].set_ylabel(chem_column)

    for i, mox_column in enumerate(df_mox.columns):
        ax[i+1+3].plot(df_mox[mox_column], c='k', label=mox_column)
        ax[i+1+3].set_ylabel(mox_column)

    for ax_ in ax:
        ax_.grid()
    for i, event_column in enumerate(df_event.columns):
        ax2 = ax[i+1+3].twinx()
        ax2.plot(df_event[event_column], c='g', label=event_column)
        ax2.set_ylabel(event_column)


    ax[-1].set_xlabel("Datetime")
    fig.align_ylabels()
    plt.show()

# Extract events and build training set
X_train = []
y_train = ['slalom', 'slalom', 'slalom', 'slalom', 'giant_slalom', 'downhill', 'giant_slalom', 'giant_slalom', 'downhill', 'downhill', 'giant_slalom', 'downhill']
events, index_events = detect_events(df)
for i, event in enumerate(events):
    features = build_featuremap(df, event, k_max=20)
    X_train.append(features)
assert len(X_train)==len(y_train)
X_train = np.array(X_train)
y = np.array(y_train)

# Train scaler
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

# Train classifier
svm = SVC(kernel='linear').fit(X_scaled, y_train)
y_pred = svm.predict(X_scaled)
# print(y_train)
# print(y_pred)



# Load testing data
testing_data = Path('data/WT1-30146_test.csv')
if not testing_data.exists():
    raise ValueError(f"{testing_data} does not exist.")
df_test = pd.read_csv(testing_data, sep=';')

# Filter and preprocess data
df_test = df_test[df_test['element']=='WT1-30146'].dropna(subset=['air_mox0'])
df_test['date'] = pd.to_datetime(df_test['date']).apply(lambda t: t.replace(tzinfo=None))

df_test = df_test[df_test['date'] > pd.to_datetime('16.01.23 12:00')]

# Set date as index
df_test = df_test.set_index('date')

# Extract events and build testing set
X_test = []
events, index_events = detect_events(df_test)
# print(events, index_events)
for i, event in enumerate(events):
    features = build_featuremap(df_test, event, k_max=20)
    X_test.append(features)
X_test = np.array(X_test)

# Transform with scaler
# scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)

# Predict with classifier
y_test_pred = svm.predict(X_test_scaled)
print("Prediction:", y_test_pred)

# exit()

if explore: 
    # Create dataframes for each measurement type
    df_pid = df_test['pid_ug_m3'].astype('float')
    df_mox = df_test[['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']].astype('float')
    df_chem = df_test[['air_NH3', 'air_H2S']].astype('float')
    df_event = df_test[['event_sensor_01', 'event_sensor_02', 'event_sensor_03', 'event_sensor_04']].astype('float')

    # Create the figure and axes
    fig, ax = plt.subplots(nrows=8, sharex=True)

    # Plot the data
    # ax[0].scatter(df.index, [1]*len(df), c=[label_colors[l] for l in df['odour']], s=2)

    ax[1].plot(df_pid, c='r', label='PID')
    ax[1].set_ylabel('PID')

    for i, chem_column in enumerate(df_chem.columns):
        ax[i+2].plot(df_chem[chem_column], c='b', label=chem_column)
        ax[i+2].set_ylabel(chem_column)

    for i, mox_column in enumerate(df_mox.columns):
        ax[i+1+3].plot(df_mox[mox_column], c='k', label=mox_column)
        ax[i+1+3].set_ylabel(mox_column)

    for ax_ in ax:
        ax_.grid()
    # for i, event_column in enumerate(df_event.columns):
    #     ax2 = ax[i+1+3].twinx()
    #     ax2.plot(df_event[event_column], c='g', label=event_column)
    #     ax2.set_ylabel(event_column)


    ax[-1].set_xlabel("Datetime")
    fig.align_ylabels()
    plt.show()
    exit()

exit()
##########
## IDEA (assuming same concentration & same duration stimulus):
## 1. Filter 
## 2. Sliding window
## 3. Label training data windows -> compose library 
## 4. Nearest Neighbor of each window of testing data with training data windows
##
#

##########
## OTHER IDEA (less strict assumption on same concentration & same duration stimulus):
## 1. Filter 
## 2. Sliding window (rather large, e.g. 5min) over training data and testing data
## 3. Extract lots of different features ROCKET-style: many many random-sized kernels for each window. Use same for training and testing.
## 4. Maybe do PCA on training data to get lower dimensional features. 
## 4. Label training data windows -> compose library 
## 5. Nearest Neighbor Classifier or SVM or whatever, of each window of testing data features with training data windows
##
#
# ChatGPT:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the rolling window size
window_size = 100

# Scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.T).T

# Initialize the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Initialize a list to store the predictions
predictions = []

# Iterate over the training data with a rolling window
for i in range(0, X_train_scaled.shape[1] - window_size):
    # Extract the current window of data
    X_window = X_train_scaled[:, i:i+window_size]
    y_window = y_train[i:i+window_size]

    # Flatten the data to 2D array for fitting the model
    X_window = X_window.reshape((-1, window_size*X_train_scaled.shape[0]))
    y_window = y_window.flatten()
    
    # Fit the k-NN model on the current window of data
    knn.fit(X_window, y_window)

    # Predict the class for the next sample
    y_pred = knn.predict(X_train_scaled[:, i+window_size].reshape(1, -1))
    predictions.append(y_pred)

# Evaluate the performance of the model
# print("Accuracy:", np.mean(pred



## again

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Create training data
X_train = np.random.rand(6, 10000)
y_train = np.random.randint(0, 3, size=(10000))

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.T)

# Define the sliding window size
window_size = 100

# Create an empty list to store the predictions
predictions = []

# Iterate over the test data with a sliding window
for i in range(0, X_test_scaled.shape[0], window_size):
    # Get the current window
    X_test_window = X_test_scaled[i:i+window_size, :]
    
    # Reshape the window to match the training data
    X_test_window = X_test_window.reshape(1, -1, 6)
    
    # Create a k-NN model
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Fit the model to the training data
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions on the current window
    window_predictions = knn.predict(X_test_window)
    
    # Append the predictions to the list
    predictions.append(window_predictions)

# Flatten the list of predictions
predictions = [item for sublist in predictions for item in sublist]

###########################
# R-NN

import torch
import torch.nn as nn

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Create training data
X_train = np.random.rand(6, 10000)
y_train = np.random.randint(0, 4, size=(10000)) # baseline + 3 different anomalies

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.T)

# Reshape the data to (batch_size, sequence_length, input_size)
X_train_scaled = X_train_scaled.reshape(-1, window_size, 6)
y_train = y_train.reshape(-1, window_size)

# Define the sliding window size
window_size = 100

# Initialize the RNN model
rnn = RNN(input_size=6, hidden_size=8, num_layers=1, num_classes=4)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Train the RNN
for epoch in range(100):
    outputs = rnn(X_train_scaled)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item():.4f}')

# Test the RNN
with torch.no_grad():
    X_test_scaled = scaler.transform(X_test.T)
    # Reshape the test data to (batch_size, sequence_length, input_size)
    X_test_scaled = X_test_scaled.reshape(-1, window_size, 6)
    
    # Make predictions on the test data
    test_predictions = rnn(X_test_scaled)
    test_predictions = test_predictions.argmax(dim=1)

# Flatten the list of predictions
test_predictions = test_predictions.flatten()

######################
import torch
import torch.nn as nn

# Define the 1D CNN model
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass input through the first convolutional layer
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.max_pool(x)
        
        # Pass through the second convolutional layer
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.max_pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layer
        x = self.fc(x)
        return x

# Create training data
X_train = np.random.rand(6, 10000)
y_train = np.random.randint(0, 4, size=(10000)) # baseline + 3 different anomalies

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.T)

# Define the sliding window size
window_size = 100

# Reshape the data to (batch_size, sequence_length, input_size)
X_train_scaled = X_train_scaled.reshape(-1, 6, window_size)
y_train = y_train.reshape(-1, window_size)

# Initialize the 1D CNN model
cnn = CNN(input_size=6, hidden_size=8, num_layers=1, num_classes=4)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# Train the 1D CNN
for epoch in range(100):
    outputs = cnn(X_train_scaled)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item():.4f}')

# Test the 1D CNN

# %%

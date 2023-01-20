import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score

def exploration(df, show_events=False, show_groundtruth=True):
    # Explore
    label_colors = {
    'Downhill': 'red', 
    'Giant Slalom': 'blue', 
    'Slalom': 'green', 
    'ambient': 'gray',
    }

    # Create dataframes for each measurement type
    df_pid = df['pid_ug_m3'].astype('float')
    df_mox = df[['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']].astype('float')
    df_chem = df[['air_NH3', 'air_H2S']].astype('float')
    df_event = df[['event_sensor_01', 'event_sensor_02', 'event_sensor_03', 'event_sensor_04']].astype('float')

    # Find peaks
    pid_1stderivative = np.gradient(df_pid.to_numpy())
    peaks_x, peaks_y = signal.find_peaks(pid_1stderivative, height=10, distance=40)
    peaks_y = peaks_y['peak_heights']

    # Create the figure and axes
    fig, ax = plt.subplots(nrows=8, sharex=True, figsize=(8,8))

    # Plot the data
    if show_groundtruth:
        ax[0].scatter(df.index, [1]*len(df), c=[label_colors[l] for l in df['odour']], label=[l for l in df['odour']], s=20, marker="|")
        ax[0].set_ylabel("Ground truth")
    
    ax[1].plot(df_pid, c='r', label='PID')
    ax[1].set_ylabel('PID')

    if show_events:
        df_pid.iloc[peaks_x].plot(style='.', ax=ax[1], c='k')

    for i, chem_column in enumerate(df_chem.columns):
        ax[i+2].plot(df_chem[chem_column], c='b', label=chem_column)
        ax[i+2].set_ylabel(chem_column)

    for i, mox_column in enumerate(df_mox.columns):
        ax[i+1+3].plot(df_mox[mox_column], c='k', label=mox_column)
        ax[i+1+3].set_ylabel(f"MOX{i+1}")

    for ax_ in ax:
        ax_.grid()
    for i, event_column in enumerate(df_event.columns):
        ax2 = ax[i+1+3].twinx()
        ax2.plot(df_event[event_column], c='g', label=event_column)
        ax2.set_ylabel(f"d/dx MOX{i+1}")
        ax2.yaxis.label.set_color('green')

    ax[-1].set_xlabel("Datetime")
    fig.align_ylabels()
    plt.show()

def detect_events(df):
    # Find peaks of PID 1st derivative
    df_pid = df['pid_ug_m3'].astype('float')
    pid_1stderivative = np.gradient(df_pid.to_numpy())
    peaks_x, peaks_y = signal.find_peaks(pid_1stderivative, height=10, distance=40)
    peaks_y = peaks_y['peak_heights']
    index_events = df.iloc[peaks_x].index
    return peaks_x, index_events

def build_featuremap(df, event):
    event_columns=['event_sensor_01', 'event_sensor_02', 'event_sensor_03', 'event_sensor_04', 'air_H2S']
    mox_columns = ['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']
    minus = -6 # 1 minute before detected onset
    plus = 36  # 6 minutes after detected onset

    # Take polyfit of MOX as features
    subset_mox = df[mox_columns].astype('float').to_numpy()
    window = subset_mox[event+minus: event+plus]
    polyfeatures = []
    for arr in window.T:
        poly = np.polyfit(range(len(arr)), arr, deg=4)
        polyfeatures.append(poly)    
    polyfeatures = np.array(polyfeatures).flatten()

     # Take window of 1st MOX derivative (aka event sensors) & air_H2S as features
    subset_events = df[event_columns].astype('float').to_numpy()
    window_events = subset_events[event+minus: event+plus]
    windowfeatures = window_events.flatten()

    # Concatenate & return
    return np.concatenate([windowfeatures, polyfeatures])

def load_training_labels(path):
    training_labels = Path(path)
    if not training_labels.exists():
        raise ValueError(f"{training_labels} does not exist.")
    labels = pd.read_csv(training_labels, sep=',')
    labels['date_start'] = pd.to_datetime(labels['Date'] + ' ' + labels['Start'])
    labels['date_end'] = pd.to_datetime(labels['Date'] + ' ' + labels['End'])
    return labels

def load_data(path, labels=None):
    data = Path(path)
    if not data.exists():
        raise ValueError(f"{data} does not exist.")
    df = pd.read_csv(data, sep=';', low_memory=False)

    # Filter and preprocess data
    df = df[df['element']=='WT1-30146'].dropna(subset=['air_mox0'])
    df['date'] = pd.to_datetime(df['date']).apply(lambda t: t.replace(tzinfo=None))
    df = df[df['date'] > pd.to_datetime('16.01.23 14:15')]

    if labels is not None:
        # Set labels to training data (just for plotting)
        df['odour'] = 'ambient'
        for _, row in labels.iterrows():
            for idx, df_row in df[(df['date'] >= row['date_start']) & (df['date'] <= row['date_end'])].iterrows():
                df.loc[idx, 'odour'] = row['Odour Type']
    # Set date as index
    df = df.set_index('date')
    return df

def get_features(df, events):
    X = []
    for i, event in enumerate(events):
        features = build_featuremap(df, event)
        X.append(features)
    X = np.array(X)
    return X

def train_validate(X_train_all, y_train_all, train_idxs, val_idxs):
    scores = []

    accuracies = []
    recalls = []
    f1_scores = []
    for train_idx, val_idx in zip(train_idxs, val_idxs):
        X_train, y_train = X_train_all[train_idx], y_train_all[train_idx]
        X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]

        # Train scaler
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
        X_scaled = scaler.transform(X_train)

        # Train classifier
        # classifier = KNeighborsClassifier().fit(X_scaled, y_train)
        classifier = SVC(kernel='linear').fit(X_scaled, y_train)
        y_pred = classifier.predict(X_scaled)

        # Validate classifier
        X_val_scaled = scaler.transform(X_val)
        y_pred = classifier.predict(X_val_scaled)
        score = classifier.score(X_val_scaled, y_val)
        scores.append(score)

        accuracies.append(accuracy_score(y_val, y_pred))
        # recalls.append(recall_score(y_val, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
        
        # print(score)
    print(f"\n Validation:")
    print(f"\t Accuracy: {np.mean(accuracies):2f} pm {np.std(accuracies):2f}")
    # print(f"\t Recall: {np.mean(recalls):2f} pm {np.std(recalls):2f}")
    print(f"\t F1 Score: {np.mean(f1_scores):2f} pm {np.std(f1_scores):2f}")
    

def train_full(X_train_all, y_train_all):
    # Train scaler
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train_all)
    X_scaled_all = scaler.transform(X_train_all)

    # Train classifier
    classifier = SVC(kernel='linear').fit(X_scaled_all, y_train_all)
    return scaler, classifier

def pc_analysis(X_train_all, y_train_all, pca=None):
    scaler = StandardScaler().fit(X_train_all)
    X_train_all = scaler.transform(X_train_all)

    if pca is None:
        pca = PCA()
        pca.fit(X_train_all.T)

    x_transformed = pca.transform(X_train_all.T).T
    print(X_train_all.shape, x_transformed.shape, y_train_all.shape)

    colors = {
        'slalom': 'blue',
        'giant_slalom': 'purple',
        'downhill': 'orange'
    }

    fig, ax = plt.subplots()
    for x, y in zip(x_transformed, y_train_all):
        ax.scatter(x[0], x[1], label=y, c=colors[y])
    plt.show()
    return pca

def test_predict(X_test, scaler, classifier):
    # Transform with scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict with classifier
    y_test_pred = classifier.predict(X_test_scaled)
    return y_test_pred

if __name__ == '__main__':
    # Run settings
    EXPLORE_TRAIN = True
    EXPLORE_TRAIN_ONSET = True
    VALIDATE = True
    TRAIN_FULL = True
    EXPLORE_TEST = True#True
    TEST = True

    # Load ground truth
    path_training_labels = 'data/Training_Dataset_ISOCSWinterSchool2023.csv'
    labels = load_training_labels(path_training_labels)

    # List labels in their occuring order (manually)
    y_train_all = np.array(['slalom', 'slalom', 'slalom', 'slalom', 'giant_slalom', 'downhill', 'giant_slalom', 'giant_slalom', 'downhill', 'downhill', 'giant_slalom', 'downhill'])

    # Load training data
    path_training = 'data/WT1-30146.csv'
    df = load_data(path_training, labels)

    # Show training data
    if EXPLORE_TRAIN:
        exploration(df, show_events=False, show_groundtruth=True)

    # Extract events 
    events, index_events = detect_events(df)

    # Show events on training data
    if EXPLORE_TRAIN_ONSET:
        exploration(df, show_events=True, show_groundtruth=True)

    # Extract features and build training set
    X_train_all = get_features(df, events)
    assert len(X_train_all)==len(y_train_all)

    # # Principal component analysis 
    # pca = pc_analysis(X_train_all, y_train_all)

    # Cross-validated training
    train_idxs = [
        [2, 7, 9, 1, 6, 8, 0, 4, 5], 
        [3, 10, 11, 1, 6, 8, 0, 4, 5], 
        [3, 10, 11, 2, 7, 9, 0, 4, 5], 
        [3, 10, 11, 2, 7, 9, 1, 6, 8]
        ]
    val_idxs = [
        [3, 10, 11], 
        [2, 7, 9], 
        [1, 6, 8], 
        [0, 4, 5]
        ]
    train_validate(X_train_all, y_train_all, train_idxs, val_idxs)

    # Training on all training data
    scaler, classifier = train_full(X_train_all, y_train_all)

    # Load testing data
    path_testing = 'data/WT1-30146_final.csv'
    df_test = load_data(path_testing)

    # Explore test data
    if EXPLORE_TEST:
        exploration(df_test, show_events=False, show_groundtruth=False)

    # Extract events 
    events, index_events = detect_events(df_test)

    # Show events on training data
    if EXPLORE_TEST:
        exploration(df_test, show_events=True, show_groundtruth=False)

    if TEST:
        # Extract features and build training set
        X_test = get_features(df_test, events)

        # Test trained model on test data
        test(X_test, scaler, classifier)
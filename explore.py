import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
training_data = Path('data/WT1-30146.csv')
if not training_data.exists():
    raise ValueError(f"{training_data} does not exist.")
df = pd.read_csv(training_data, sep=';')

# Filter and preprocess data
df = df[df['element']=='WT1-30146'].dropna(subset=['air_mox0'])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Create dataframes for each measurement type
df_pid = df['pid_ug_m3'].astype('float')
df_mox = df[['air_mox0', 'air_mox1', 'air_mox2', 'air_mox3']].astype('float')
df_chem = df[['air_COV', 'air_NH3', 'air_H2S']].astype('float')

# Create the figure and axes
fig, ax = plt.subplots(nrows=8, sharex=True)

# Plot the data
ax[0].plot(df_pid, c='r', label='PID')
ax[0].set_ylabel('PID')

for i, chem_column in enumerate(df_chem.columns):
    ax[i+1].plot(df_chem[chem_column], c='b', label=chem_column)
    ax[i+1].set_ylabel(chem_column)

for i, mox_column in enumerate(df_mox.columns):
    ax[i+1+3].plot(df_mox[mox_column], c='k', label=mox_column)
    ax[i+1+3].set_ylabel(mox_column)

ax[-1].set_xlabel("Datetime")
fig.align_ylabels()
plt.show()
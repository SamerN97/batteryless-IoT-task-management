import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

downSampleFactor = 3
antwerpDownSampleFactor = 6  # No downsampling needed; already at 30s resolution
# 30-second resolution = 2880 steps per day
STEPS_PER_DAY = 2880 

# --- Read First Dataset (Validation) ---
dataList1 = pd.read_csv('shuffled_solar_validation_dataset.csv', sep=',')
dataList1 = dataList1.iloc[:, 0].to_numpy()
# Downsample from 10s to 30s resolution
dataList1 = dataList1[::downSampleFactor] # Downsample from 10s to 30s resolution

# --- Read Second Dataset (Antwerp) ---
dataList2 = pd.read_csv('antwerp_solar_data_new_conversion.csv', sep=',')
dataList2 = dataList2.iloc[:, 0].to_numpy()
dataList2 = dataList2[::antwerpDownSampleFactor]  # Downsample from 5s to 30s resolution

# Create X-axes in units of Days for both datasets
x_days1 = np.arange(len(dataList1)) / STEPS_PER_DAY
x_days2 = np.arange(len(dataList2)) / STEPS_PER_DAY

plt.figure(figsize=(8, 4))

# Plot both datasets with transparency to see overlaps
plt.plot(x_days1, dataList1, color='green', linewidth=1.5, alpha=0.8, label='Validation Data ($I_H$)')
plt.plot(x_days2, dataList2, color='blue', linewidth=1.5, alpha=0.7, label='Antwerp Data ($I_H$)')

plt.xlabel('Time (Days)', fontsize=14)
plt.ylabel('Harvesting Current (A)', fontsize=14) 

# Dynamically set ticks and limits based on whichever dataset spans more days
max_days = max(x_days1.max(), x_days2.max())

plt.xticks(np.arange(0, max_days + 1, 5), fontsize=12) 
plt.yticks(fontsize=12)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim(0, max_days) 

plt.legend(loc='upper right', fontsize=10, framealpha=0.85)

filepath = os.path.join('validation_solar_data', 'validation_plot_days_overlay.pdf')
os.makedirs(os.path.dirname(filepath), exist_ok=True)

plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved trend plot with Days on X-axis: {filepath}")
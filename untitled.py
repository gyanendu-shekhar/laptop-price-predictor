import numpy as np
import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('laptop_data.csv')

print("Brands:", df['Brand'].unique())
print("Types:", df['TypeName'].unique())
print("Processors:", df['Cpu'].unique())
print("RAM Options:", df['Ram'].unique())
print("GPU Options:", df['Gpu'].unique())
print("HDD Options:", df['HDD'].unique())
print("SSD Options:", df['SSD'].unique())
print("Screen Sizes:", df['Inches'].unique())
print("Resolutions:", df['ScreenResolution'].unique())
print("Touchscreen Options:", df['Touchscreen'].unique())
print("IPS Panel Options:", df['Ips'].unique())

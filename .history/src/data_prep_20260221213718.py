import pandas as pd
import numpy as np
import os

# Full path to Data folder (capital D)
data_dir = r'D:\L4S1\ML\Assignment\impl\Data'
data_path = os.path.join(data_dir, 'youtube_data.csv')

df = pd.read_csv(data_path)
print('Loaded! Columns:', df.columns.tolist())
print('Shape:', df.shape)

# Inspect first rows
print(df.head())

# Engineer target - UPDATE COLUMNS TO MATCH YOUR CSV
# Common Kaggle YT cols: subscribers, views, videos, etc. Adjust below!
df['years_active'] = 3.0  # Placeholder; use real if available (e.g., end_date - start_date)
df['total_views'] = df.get('views', df.get('total_views', 0))  # Map common names
df['subscribers'] = df.get('subscribers', df.get('subs', 0))
df['growth_rate'] = df['total_views'] / df['subscribers'].clip(lower=1) / df['years_active']
df['high_performing'] = (df['growth_rate'] > df['growth_rate'].quantile(0.7)).astype(int)

print('Target balance:\n', df['high_performing'].value_counts(normalize=True))

# Save to Data folder
output_path = os.path.join(data_dir, 'processed_yt.csv')
df.to_csv(output_path, index=False)
print(f'Saved processed_yt.csv to {data_dir}')

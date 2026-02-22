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
# REPLACE engineering section with:
df['total_views'] = df['view_count']
df['subscribers'] = df['subscriber_count']
df['video_count'] = df['video_count']  # Use real column
df['years_active'] = 3.0  # Placeholder (all same time)

# BETTER target: Use quantile on video_count/subscribers ratio + views
df['engagement_proxy'] = df['view_count'] / df['subscriber_count'].clip(lower=1)
df['activity_score'] = df['video_count'] / df['years_active']
df['growth_rate'] = df['engagement_proxy'] * df['activity_score']
df['high_performing'] = (df['growth_rate'] > df['growth_rate'].quantile(0.75)).astype(int)  # 25% positive

print('Target balance:\n', df['high_performing'].value_counts(normalize=True))

# Save to Data folder
output_path = os.path.join(data_dir, 'processed_yt.csv')
df.to_csv(output_path, index=False)
print(f'Saved processed_yt.csv to {data_dir}')

import pandas as pd
import numpy as np

df = pd.read_csv('youtube_data.csv')  

df['growth_rate'] = df['total_views'] / df['subscribers'].clip(lower=1) / df.get('years_active', 2)  # Assume or compute
df['high_performing'] = (df['growth_rate'] > df['growth_rate'].quantile(0.7)).astype(int)
print(df['high_performing'].value_counts(normalize=True))  # Check balance
df.to_csv('processed_yt.csv', index=False)

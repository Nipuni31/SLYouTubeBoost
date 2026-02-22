import pandas as pd
path = r'D:\L4S1\ML\Assignment\impl\Data\processed_yt.csv'
print('Loading', path)
df = pd.read_csv(path)
cols = ['view_count','subscriber_count','video_count','years_active','total_views','subscribers','growth_rate','engagement_proxy','activity_score','high_performing']
print('\nColumns present:')
print([c for c in cols if c in df.columns])
print('\nFirst 5 rows (selected cols):')
print(df[cols].head())
print('\nPositive class (high_performing==1) descriptive stats:')
print(df[df['high_performing']==1][['growth_rate','engagement_proxy','activity_score']].describe())

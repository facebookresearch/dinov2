import pandas as pd

train_csv = 'cleaned_train_all_descriptions (13).csv'
val_csv = 'cleaned_val_all_descriptions (10).csv'

train_df = pd.read_csv(train_csv, delimiter='\t', quotechar='"', engine='python')
val_df = pd.read_csv(val_csv, delimiter='\t', quotechar='"', engine='python')

train_labels = set(train_df['description'].unique())
filtered_val_df = val_df[val_df['description'].isin(train_labels)]

print(f"Original val rows: {len(val_df)}, Filtered val rows: {len(filtered_val_df)}")
filtered_val_df.to_csv('filtered_' + val_csv, sep='\t', index=False)
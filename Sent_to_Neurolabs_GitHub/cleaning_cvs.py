import os
import pandas as pd

csv_path = '/home/sbruno/Documents/Neurolabs-dinov2/dataset_syn_abi/dataset_syn/val_all_descriptions (10).csv'            
img_dir = '/home/sbruno/Documents/Neurolabs-dinov2/dataset_syn_abi/dataset_syn/img'             

df = pd.read_csv(csv_path, delimiter='\t', quotechar='"', engine='python')
missing_mask = ~df['image_path'].apply(lambda x: os.path.isfile(os.path.join(img_dir, x.strip())))
missing_count = missing_mask.sum()

print(f"Number of missing image files: {missing_count}")
print(f"Total images listed: {len(df)}")
print(f"Number of valid images: {len(df) - missing_count}")


clean_df = df[~missing_mask]
clean_df.to_csv('cleaned_' + os.path.basename(csv_path), sep='\t', index=False)
print(f"Cleaned CSV written to cleaned_{os.path.basename(csv_path)}")
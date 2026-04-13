import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths
csv_path = "dataset/metadata.csv"
image_dir = "dataset/images"

train_dir = "dataset/train"
val_dir = "dataset/val"

# Load metadata
df = pd.read_csv(csv_path)

# OPTIONAL: Reduce classes (for faster demo)
selected_classes = ['nv', 'mel', 'bcc']  
df = df[df['dx'].isin(selected_classes)]

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

def copy_images(dataframe, target_dir):
    for _, row in dataframe.iterrows():
        label = row['dx']
        img_name = row['image_id'] + ".jpg"

        src = os.path.join(image_dir, img_name)
        dst_dir = os.path.join(target_dir, label)

        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, img_name)

        if os.path.exists(src):
            shutil.copy(src, dst)

# Run copy
copy_images(train_df, train_dir)
copy_images(val_df, val_dir)

print("✅ Dataset ready!")
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, tabular_inputs, labels, batch_size=32, target_size=(224, 224)):
        self.image_paths = image_paths
        self.tabular_inputs = tabular_inputs
        self.labels = labels
        self.batch_size = int(batch_size)
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_tabular = self.tabular_inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [load_and_preprocess_image(path, self.target_size) for path in batch_paths]
        return (np.array(batch_images), np.array(batch_tabular)), np.array(batch_labels)

def load_ham10000(data_dir="data", test_size=0.2, random_state=None):
    img_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
    img_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")

    df = pd.read_csv(metadata_path)

    # Remove unknown genders and encode binary : male = 1, female = 2
    df = df[df['sex'].isin(['male', 'female'])]
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})

    # Fill missing ages with median
    df['age'] = df['age'].fillna(df['age'].median())

    # Create 5-class age bins: ≤30, 31–45, 46–60, 61–75, ≥76
    df['age_bin'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 75, 120],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    # Encode other columns
    df['localization'] = LabelEncoder().fit_transform(df['localization'].astype(str))
    df['label'] = LabelEncoder().fit_transform(df['dx'])

    # Map images
    image_paths = []
    for img_name in df['image_id']:
        path1 = os.path.join(img_dir1, f"{img_name}.jpg")
        path2 = os.path.join(img_dir2, f"{img_name}.jpg")
        full_path = path1 if os.path.exists(path1) else path2
        image_paths.append(full_path)

    # Features for image model
    features = df[['age', 'sex', 'localization']].values
    labels = df['label'].values
    sensitive_attrs = df[['sex', 'age_bin']].values

    # Metadata for tabular model
    metadata_cols = ['age', 'sex', 'localization']
    metadata_tabular = df[metadata_cols].values


    img_train, img_test, feat_train, feat_test, label_train, label_test, sens_train, sens_test, meta_train, meta_test = train_test_split(
        image_paths, features, labels, sensitive_attrs, metadata_tabular,
        test_size=test_size, random_state=random_state, stratify=labels
    )

    return {
        'image_client': {
            'train': (np.array(img_train), np.array(feat_train), np.array(label_train), np.array(sens_train)),
            'test': (np.array(img_test), np.array(feat_test), np.array(label_test), np.array(sens_test))
        },
        'vertical_client': {
            'train': (np.array(meta_train), np.array(label_train)),
            'test': (np.array(meta_test), np.array(label_test)),
        }
    }

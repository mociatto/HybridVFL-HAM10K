import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Image loading helper
def load_and_preprocess_image(img_path, target_size=(218, 178)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, tabular_inputs, labels, batch_size=32, target_size=(218, 178)):
        self.image_paths = image_paths
        self.tabular_inputs = tabular_inputs
        self.labels = labels
        self.batch_size = int(batch_size)  # Ensure batch_size is an integer
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_tabular = self.tabular_inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [load_and_preprocess_image(path, self.target_size) for path in batch_paths]
        return (np.array(batch_images), np.array(batch_tabular)), np.array(batch_labels)
        
def load_data(data_root_path):
    img_dir = os.path.join(data_root_path, 'img_align_celeba', 'img_align_celeba')
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    attr_file = os.path.join(data_root_path, 'list_attr_celeba.csv')
    attr_df = pd.read_csv(attr_file)
    attributes = {}
    for column in attr_df.columns:
        if column != 'image_id':
            attributes[column] = attr_df[column].values
    return image_files, attributes, img_dir

def data_partition(image_files, attributes, img_dir, data_root_path, LABEL_NAME='Smiling', P1_NAME=['Straight_Hair', 'Wavy_Hair']):
    eval_file = os.path.join(data_root_path, 'list_eval_partition.csv')
    eval_df = pd.read_csv(eval_file)
    image_to_idx = {img: idx for idx, img in enumerate(image_files)}
    train_indices = [image_to_idx[img_id] for img_id in eval_df[eval_df['partition'] == 0]['image_id']]
    val_indices = [image_to_idx[img_id] for img_id in eval_df[eval_df['partition'] == 1]['image_id']]
    test_indices = [image_to_idx[img_id] for img_id in eval_df[eval_df['partition'] == 2]['image_id']]
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    P1_attribute = np.concatenate([to_categorical(attributes[a]) for a in P1_NAME], axis=-1)
    train_data = [
        [os.path.join(img_dir, f) for f in np.array(image_files)[train_indices]],
        P1_attribute[train_indices]
    ]
    train_label = to_categorical(attributes[LABEL_NAME][train_indices])
    train_attr = [to_categorical(attributes['Male'][train_indices]), to_categorical(attributes['Young'][train_indices])]
    val_data = [
        [os.path.join(img_dir, f) for f in np.array(image_files)[val_indices]],
        P1_attribute[val_indices]
    ]
    val_label = to_categorical(attributes[LABEL_NAME][val_indices])
    val_attr = [to_categorical(attributes['Male'][val_indices]), to_categorical(attributes['Young'][val_indices])]
    test_data = [
        [os.path.join(img_dir, f) for f in np.array(image_files)[test_indices]],
        P1_attribute[test_indices]
    ]
    test_label = to_categorical(attributes[LABEL_NAME][test_indices])
    test_attr = [to_categorical(attributes['Male'][test_indices]), to_categorical(attributes['Young'][test_indices])]
    return train_data, train_label, train_attr, val_data, val_label, val_attr, test_data, test_label, test_attr
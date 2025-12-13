from loading_data import load_images, LoadingType
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

class RoadSurfaceClassifier:
    def load_processed_data(self):
        """Load image paths efficiently without loading all images into RAM"""
        print("Loading image paths...")
        images = load_images(LoadingType.PROCESSED)

        image_paths: list[str] = []
        labels: list[str] = []
        for img_data in images:
            image_paths.append(img_data.image_path)
            labels.append(img_data.category.value)

        # good => 0
        # bad  => 1
        # satif => 2
        # verybad => 3

        self.class_names = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        labels_int = [label_to_idx[label] for label in labels]

        self.num_classes = len(self.class_names)

        print(f"Found {len(image_paths)} images")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution: {np.bincount(labels_int)}")

        return image_paths, labels_int

    def split_data(self, image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

        paths_temp, paths_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_ratio, random_state=42, stratify=labels
        )

        val_ratio /= train_ratio + val_ratio
        paths_train, paths_val, y_train, y_val = train_test_split(
            paths_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img)
            img = tf.cast(img, tf.float32) / 255.0
            label_one_hot = tf.one_hot(label, self.num_classes)
            return img, label_one_hot

        train_ds = tf.data.Dataset.from_tensor_slices((paths_train, y_train))
        train_ds = train_ds.map(
            load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((paths_val, y_val))
        val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((paths_test, y_test))
        test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds


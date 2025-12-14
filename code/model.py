from loading_data import Category, load_images, LoadingType
import numpy as np
import tensorflow as tf
from pathlib import Path
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)

dirname = Path(__file__).parent

class RoadSurfaceClassifier:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.num_classes = len(Category)
        self.class_names = sorted([c.value for c in Category])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

    def load_datasets(self):
        """Load train, validation, and test datasets from pre-splitted folders."""
        print("Loading datasets...")

        def create_dataset(split_name: str):
            images = load_images(LoadingType.SPLITTED, split_name)
            image_paths = [img.image_path for img in images]
            labels_int = [self.label_to_idx[img.category.value] for img in images]

            print(f"Found {len(image_paths)} images for {split_name} split.")
            
            ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_int))
            ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            return ds

        train_ds_raw = create_dataset('train')
        val_ds_raw = create_dataset('val')
        test_ds_raw = create_dataset('test')
        
        train_ds = train_ds_raw.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds_raw.batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds_raw.batch(32).prefetch(tf.data.AUTOTUNE)

        print(f"\nClasses: {self.class_names}")
        return train_ds, val_ds, test_ds

    def _load_and_preprocess(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = tf.cast(img, tf.float32) / 255.0
        label_one_hot = tf.one_hot(label, self.num_classes)
        return img, label_one_hot

    def build_model(self):
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=self.input_shape),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),

                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        print("\nModel Architecture:")
        self.model.summary()

        return self.model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )
        print("\nModel compiled successfully!")

    def train(self, train_ds, val_ds, epochs=40):
        model_dir = dirname / "../models"
        model_dir.mkdir(exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(model_dir / "best_road_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ]

        print(f"\nStarting training for {epochs} epochs...")

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        print("\nTraining completed!")
        return self.history

    def evaluate(self, test_ds):
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_prec, test_recall = self.model.evaluate(test_ds)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        y_pred = self.model.predict(test_ds)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        y_true_classes = np.argmax(y_test, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.class_names))

        return y_true_classes, y_pred_classes

    def plot_training_history(self):
        if not hasattr(self, 'history') or self.history is None:
            print("No training history available!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = [('accuracy', 'val_accuracy'), ('loss', 'val_loss'), 
                   ('precision', 'val_precision'), ('recall', 'val_recall')]
        titles = ['Model Accuracy', 'Model Loss', 'Model Precision', 'Model Recall']

        for i, (ax, (train_metric, val_metric)) in enumerate(zip(axes.flatten(), metrics)):
            ax.plot(self.history.history[train_metric], label=f'Train {train_metric.capitalize()}')
            ax.plot(self.history.history[val_metric], label=f'Val {val_metric.capitalize()}')
            ax.set_title(titles[i])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(titles[i].split()[-1])
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plots_dir = dirname / "../results"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches="tight")
        print(f"\nTraining history plot saved to {plots_dir / 'training_history.png'}")
        plt.close(fig)

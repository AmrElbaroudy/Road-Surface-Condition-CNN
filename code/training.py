from loading_data import load_images, LoadingType
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

dirname = Path(__file__).parent

class RoadSurfaceClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes

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
    def build_model(self):
        self.model = models.Sequential(
            [
                # First convolutional block
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=self.input_shape,
                ),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Fourth convolutional block (deeper for 224x224 input)
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                # Dense layers
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        print("\nModel Architecture:")
        self.model.summary()

        return self.model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate), # type: ignore
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )
        print("\nModel compiled successfully!")

    def train(self, train_ds, val_ds, epochs=40):
        # Create model directory
        model_dir = dirname / "../models"
        model_dir.mkdir(exist_ok=True)

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                filepath=str(model_dir / "best_road_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
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
        test_loss, test_acc, test_prec, test_recall = self.model.evaluate(
            test_ds, 
        )

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # * only uses X part in test_ds
        y_pred = self.model.predict(test_ds)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
        y_true_classes = np.argmax(y_test, axis=1)

        print("\nClassification Report:")
        print(
            classification_report(
                y_true_classes, y_pred_classes, target_names=self.class_names
            )
        )

        return y_true_classes, y_pred_classes

    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            print("No training history available!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history["accuracy"], label="Train Accuracy")
        axes[0, 0].plot(self.history.history["val_accuracy"], label="Val Accuracy")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history["loss"], label="Train Loss")
        axes[0, 1].plot(self.history.history["val_loss"], label="Val Loss")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history["precision"], label="Train Precision")
        axes[1, 0].plot(self.history.history["val_precision"], label="Val Precision")
        axes[1, 0].set_title("Model Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history["recall"], label="Train Recall")
        axes[1, 1].plot(self.history.history["val_recall"], label="Val Recall")
        axes[1, 1].set_title("Model Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save plot
        plots_dir = dirname / "../results"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches="tight")
        print(f"\nTraining history plot saved to {plots_dir / 'training_history.png'}")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save plot
        plots_dir = (dirname / "../results").resolve()
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import keras
from model import RoadSurfaceClassifier
import numpy as np

dirname = Path(__file__).parent

def plot_accuracy_curve(history_df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    output_path = output_dir / "accuracy_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Accuracy curve saved to {output_path}")

def plot_loss_curve(history_df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    output_path = output_dir / "loss_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {output_path}")

def plot_confusion_matrix_custom(y_true, y_pred, class_names, output_dir):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    output_path = output_dir / "confusion_matrix_visual.png" # Different name to avoid conflict
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_sample_predictions(test_ds, model, class_names, output_dir):
    plt.figure(figsize=(15, 15))

    all_images = []
    all_true_labels = []
    # Collect more samples than just one batch to ensure variety
    # Take a few batches, or iterate until we have enough samples
    num_batches_to_collect = 5
    for images, labels in test_ds.take(num_batches_to_collect):
        all_images.append(images.numpy())
        all_true_labels.append(np.argmax(labels.numpy(), axis=1))

    all_images = np.concatenate(all_images, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Ensure we don't try to sample more than available
    num_samples_to_plot = min(16, len(all_images))
    
    # Randomly select indices
    random_indices = np.random.choice(len(all_images), num_samples_to_plot, replace=False)

    # Get predictions for the randomly selected images
    selected_images = all_images[random_indices]
    selected_true_labels = all_true_labels[random_indices]

    predictions = model.predict(selected_images)
    pred_labels = np.argmax(predictions, axis=1)

    for i in range(num_samples_to_plot):
        plt.subplot(4, 4, i + 1)
        plt.imshow(selected_images[i])
        true_class = class_names[selected_true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        plt.title(f"True: {true_class}\nPred: {pred_class}",
                    color="green" if true_class == pred_class else "red")
        plt.axis("off")

    plt.tight_layout()
    output_path = output_dir / "sample_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {output_path}")

def plot_classification_report(y_true, y_pred, class_names, output_dir):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(12, len(report_df) * 0.5))
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap="viridis", fmt=".2f")
    plt.title("Classification Report")
    output_path = output_dir / "classification_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Classification report plot saved to {output_path}")
    return report_df


def main():
    """
    Main function to generate and save visualizations.
    """
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    # --- Plotting from history ---
    history_path = project_root / "results/training_history.csv"
    if not history_path.exists():
        print(f"Training history file not found at {history_path}")
        print("Please run the training script 'uv run code/training.py' first.")
        return
        
    history_df = pd.read_csv(history_path)
    plot_accuracy_curve(history_df, output_dir)
    plot_loss_curve(history_df, output_dir)

    # --- Plotting from model evaluation ---
    model_path = project_root / "models/best_road_model.keras"
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        return

    classifier = RoadSurfaceClassifier()
    # We only need the test dataset for these visualizations
    _, _, test_ds = classifier.load_datasets()
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    # Re-compile model to restore metrics
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )
    print("Model loaded and compiled successfully.")

    y_pred_encoded = model.predict(test_ds)
    y_pred = np.argmax(y_pred_encoded, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    class_names = classifier.class_names

    plot_confusion_matrix_custom(y_true, y_pred, class_names, output_dir)
    plot_sample_predictions(test_ds, model, class_names, output_dir)
    plot_classification_report(y_true, y_pred, class_names, output_dir)

    print("\nAll visualizations saved successfully!")

if __name__ == "__main__":
    main()
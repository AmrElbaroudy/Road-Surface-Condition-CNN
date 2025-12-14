from model import RoadSurfaceClassifier
from visualizations import plot_training_history
import tensorflow as tf
from pathlib import Path
import pandas as pd

def main():
    classifier = RoadSurfaceClassifier(input_shape=(224, 224, 3))

    train_ds, val_ds, test_ds = classifier.load_datasets()

    classifier.build_model()
    classifier.compile_model(learning_rate=0.001)

    history = classifier.train(train_ds, val_ds, epochs=40)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_csv_path = Path(__file__).parent.parent / "results/training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    plot_training_history(history)

    _, _ = classifier.evaluate(test_ds)

    print("\n" + "=" * 42)
    print("Training pipeline completed successfully!")
    print("=" * 42)


if __name__ == "__main__":
    main()

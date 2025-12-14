from pathlib import Path
import keras
from model import RoadSurfaceClassifier

def main():
    classifier = RoadSurfaceClassifier()

    # Load the datasets, but we only need the test set for evaluation
    _, _, test_ds = classifier.load_datasets()

    model_path = Path(__file__).parent / "../models/best_road_model.keras"
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        print("Please train the model first by running 'uv run code/training.py'")
        return

    print(f"Loading model from {model_path}...")
    # It's important to compile the model after loading, especially with custom metrics
    classifier.model = keras.models.load_model(model_path)
    classifier.compile_model(learning_rate=0.001) # Re-compile to restore metrics, etc.
    print("Model loaded and compiled successfully.")

    classifier.evaluate(test_ds)

if __name__ == "__main__":
    main()

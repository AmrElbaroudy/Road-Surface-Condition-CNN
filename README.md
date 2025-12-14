# Road-Surface-Condition-CNN

This project uses a Convolutional Neural Network (CNN) to classify road surface conditions from images. The model can identify four categories: good, poor, satisfactory, and very poor.

## Dataset

The dataset used for this project can be found on Kaggle:
[Road Damage Classification and Assessment](https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment/data)

Download the dataset and place it in a way that the scripts can access it. The raw data should be placed in `data_set/raw`, if you want to train the model.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Note: the pre-trained model is uploaded [here](https://drive.google.com/file/d/1Zo_j-OSVwYh45CVOuFx_lmQ_De_mFaC_/view?usp=drive_link), download it and put it in `models` folder, after that you should have `models/best_road_model.keras` file in your local clone,with 98% accuracy, you can use `code/inference.py` to test the model via GUI, they will read from `models` directory.
The model is not committed to git because it's large (150mb)
You can find some sample test images to test from here `sample_test_data/`

The project is divided into several scripts, each responsible for a specific part of the workflow.

### 1. Data Preprocessing

Before training the model, you need to preprocess the raw image data. This includes resizing, augmentation (brightness, contrast, rotation, flip), and organizing the images into a processed dataset.

To run the preprocessing script:

```bash
python code/prepare_data.py
```

This will take the raw images from `data_set/raw` and save the processed images in `data_set/processed`.

### 2. Training the Model

Once the data is preprocessed, you can train the CNN model. The training script will load the processed data, build the model, and train it. The best model will be saved.

To start the training process:

```bash
python code/training.py
```

The trained model will be saved as `models/best_road_model.keras`. Training logs are saved in `logs/training.txt`, and the training history is saved in `results/training_history.csv`.

### 3. Evaluating the Model

After training, you can evaluate the model's performance on the test set. The evaluation script will generate a classification report and a confusion matrix.

To evaluate the model:

```bash
python code/evaluation.py
```

The evaluation report is saved in `logs/evaluation.txt`, and visualizations such as the accuracy and loss curves, and the confusion matrix are saved in the `results/` directory.

### 4. Running the GUI Application

An interactive GUI application is available to classify single road surface images.

To run the application:

```bash
python code/inference.py
```

This will open a window where you can select an image, and the application will display the predicted classification with probabilities for each category.

## Results

The following results and visualizations are generated during the training and evaluation process and can be found in the `results/` directory:

- `accuracy_curve.png`: Model accuracy over epochs.
- `loss_curve.png`: Model loss over epochs.
- `confusion_matrix_visual.png`: Confusion matrix of the model's predictions.
- `classification_report.png`: Classification report showing precision, recall, and F1-score.
- `sample_predictions.png`: Example predictions on test images.
- `training_history.csv`: CSV file with the training history.

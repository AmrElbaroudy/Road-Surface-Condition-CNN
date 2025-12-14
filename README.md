# Road-Surface-Condition-CNN

## 1. project description

=======

Road surface conditions play a crucial role in driving safety, transportation efficiency, and infrastructure management. Images captured by vehicle-mounted cameras or roadside monitoring systems often contain clear visual cues that indicate surface quality such as wetness, cracks, sand, snow, or potholes.

---

## 2. dataset link

This dataset contains images of roads with damages.
We classify the damages of roads into 4 categories such as good, poor, satisfactory, very poor according to their extent of damage.

**Dataset Link:**
https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment/data

---

## 3. How to Install Dependencies

---

### loading_data.py

**Purpose:**  
This file is responsible for loading road surface images from disk and organizing them
into structured objects that can be used for training and evaluation.

**Key Components:**

- **Category (Enum):**  
  Defines the road surface classes such as `good`, `poor`, `satisfactory`,
  and `very_poor`.

- **Image (Dataclass):**  
  Represents a single image with its file path and corresponding category.

- **LoadingType (Enum):**  
  Specifies whether images are loaded from the raw dataset or the processed dataset.

**Main Function:**

`load_images(type: LoadingType)`

**Process:**

- Determines the dataset directory based on the loading type
- Extracts image paths and their categories

**Output:**

- A list of `Image` objects containing image paths and labels

---

### prepare_data.py

**Purpose:**  
This file performs image preprocessing and data augmentation to generate multiple variations of the same image.
This help the model learn different possible conditions of the road surface and prevent overfitting, ensuring better generalization.

**Key Components:**

- **ProcessingImage (Dataclass):**  
  Stores the image matrix, path, and category label.

**Main Functions:**

- `read_images()`  
   Reads images using OpenCV.

- `change_resolution()`  
  Resizes all images to a fixed size (224×224).

- `variate_brightness_contrast()`  
  Applies random brightness and contrast variations.

- `variate_rotation()`  
  Rotates images by small angles for augmentation.

- `variate_flip()`  
  Applies horizontal flipping randomly.

- `save_images()`  
  Saves processed images into class-based folders.

**Output:**

- Augmented and resized dataset stored in `data_set/processed`

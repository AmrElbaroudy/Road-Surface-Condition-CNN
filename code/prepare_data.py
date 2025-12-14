from dataclasses import dataclass
from cv2.typing import MatLike
from loading_data import Category, Image, load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path
import uuid
import random
import shutil
from sklearn.model_selection import train_test_split
from typing import List

dirname = Path(__file__).parent

@dataclass
class ProcessingImage:
    path: str
    mat: MatLike
    category: Category

def read_images(images: List[Image]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        read_img = cv.imread(img.image_path)
        if read_img is None:
            continue
        result.append(ProcessingImage(mat=read_img, category=img.category, path=img.image_path))
    return result

def change_resolution(images: List[ProcessingImage], size=(224, 224)) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        resized_img = cv.resize(img.mat, size, interpolation=cv.INTER_AREA)
        result.append(ProcessingImage(mat=resized_img, category=img.category, path=img.path))
    return result

def variate_brightness_contrast(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        for _ in range(5):
            alpha = random.uniform(1.0, 1.3)
            beta = random.randint(0, 30)
            adjusted = cv.convertScaleAbs(img.mat, alpha=alpha, beta=beta)
            result.append(ProcessingImage(mat=adjusted, category=img.category, path=img.path))
    return result

def variate_rotation(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        for angle in [-10, -5, 5, 10]:
            if random.random() < 0.9:
                continue
            (h, w) = img.mat.shape[:2]
            center = (w // 2, h // 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv.warpAffine(img.mat, M, (w, h))
            result.append(ProcessingImage(mat=rotated, category=img.category, path=img.path))
    return result

def variate_flip(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        if random.random() < 0.9:
            continue
        flipped = cv.flip(img.mat, 1)
        result.append(ProcessingImage(mat=flipped, category=img.category, path=img.path))
    return result

def save_images_to_split(imgs: List[ProcessingImage], split_name: str) -> None:
    for image in imgs:
        base_name = os.path.basename(image.path).split('.')[0]
        file_name = f"{base_name}_{uuid.uuid4()}.jpg"
        category_val = image.category.value
        folder_path = (dirname / f"../data_set/splitted/{split_name}" / category_val).resolve()
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        cv.imwrite(save_path, image.mat)

def main():
    processed_path = dirname / "../data_set/processed"
    if processed_path.exists():
        print(f"Removing old processed directory: {processed_path}")
        shutil.rmtree(processed_path)

    splitted_path = dirname / "../data_set/splitted"
    if splitted_path.exists():
        print(f"Removing old splitted directory: {splitted_path}")
        shutil.rmtree(splitted_path)
    
    print("Loading raw images...")
    all_images = load_images(LoadingType.RAW)
    labels = [img.category for img in all_images]

    # Split data (80% train, 10% val, 10% test)
    # First, separate out the test set (10%)
    train_val_imgs, test_imgs, train_val_labels, _ = train_test_split(
        all_images, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Next, split the remainder into training (80%) and validation (10%)
    # val_size should be 0.1 / 0.9 = 0.1111...
    train_imgs, val_imgs, _, _ = train_test_split(
        train_val_imgs, train_val_labels, test_size=(0.1/0.9), random_state=42, stratify=train_val_labels
    )

    print(f"Total images: {len(all_images)}")
    print(f"Training set size: {len(train_imgs)}")
    print(f"Validation set size: {len(val_imgs)}")
    print(f"Test set size: {len(test_imgs)}")

    # Process and save training data with augmentation
    print("\nProcessing training data with augmentation...")
    train_proc_imgs = read_images(train_imgs)
    train_proc_imgs = change_resolution(train_proc_imgs)
    train_proc_imgs = variate_brightness_contrast(train_proc_imgs)
    train_proc_imgs = variate_rotation(train_proc_imgs)
    train_proc_imgs = variate_flip(train_proc_imgs)
    save_images_to_split(train_proc_imgs, "train")
    print(f"Saved {len(train_proc_imgs)} augmented training images.")

    # Process and save validation data (no augmentation)
    print("\nProcessing validation data...")
    val_proc_imgs = read_images(val_imgs)
    val_proc_imgs = change_resolution(val_proc_imgs)
    save_images_to_split(val_proc_imgs, "val")
    print(f"Saved {len(val_proc_imgs)} validation images.")

    # Process and save test data (no augmentation)
    print("\nProcessing test data...")
    test_proc_imgs = read_images(test_imgs)
    test_proc_imgs = change_resolution(test_proc_imgs)
    save_images_to_split(test_proc_imgs, "test")
    print(f"Saved {len(test_proc_imgs)} test images.")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
from dataclasses import dataclass
from cv2.typing import MatLike
from loading_data import Category, Image, load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path
import uuid
import random
import shutil
from sklearn.model_selection import train_test_split
from typing import List

dirname = Path(__file__).parent

@dataclass
class ProcessingImage:
    path: str
    mat: MatLike
    category: Category

def read_images(images: List[Image]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        read_img = cv.imread(img.image_path)
        if read_img is None:
            continue
        result.append(ProcessingImage(mat=read_img, category=img.category, path=img.image_path))
    return result

def change_resolution(images: List[ProcessingImage], size=(224, 224)) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        resized_img = cv.resize(img.mat, size, interpolation=cv.INTER_AREA)
        result.append(ProcessingImage(mat=resized_img, category=img.category, path=img.path))
    return result

def variate_brightness_contrast(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        for _ in range(5):
            alpha = random.uniform(1.0, 1.3)
            beta = random.randint(0, 30)
            adjusted = cv.convertScaleAbs(img.mat, alpha=alpha, beta=beta)
            result.append(ProcessingImage(mat=adjusted, category=img.category, path=img.path))
    return result

def variate_rotation(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        for angle in [-10, -5, 5, 10]:
            if random.random() < 0.9:
                continue
            (h, w) = img.mat.shape[:2]
            center = (w // 2, h // 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv.warpAffine(img.mat, M, (w, h))
            result.append(ProcessingImage(mat=rotated, category=img.category, path=img.path))
    return result

def variate_flip(images: List[ProcessingImage]) -> List[ProcessingImage]:
    result: List[ProcessingImage] = []
    for img in images:
        result.append(ProcessingImage(mat=img.mat, category=img.category, path=img.path))
        if random.random() < 0.9:
            continue
        flipped = cv.flip(img.mat, 1)
        result.append(ProcessingImage(mat=flipped, category=img.category, path=img.path))
    return result

def save_images_to_split(imgs: List[ProcessingImage], split_name: str) -> None:
    for image in imgs:
        base_name = os.path.basename(image.path).split('.')[0]
        file_name = f"{base_name}_{uuid.uuid4()}.jpg"
        category_val = image.category.value
        folder_path = (dirname / f"../data_set/splitted/{split_name}" / category_val).resolve()
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        cv.imwrite(save_path, image.mat)

def main():
    processed_path = dirname / "../data_set/processed"
    if processed_path.exists():
        print(f"Removing old processed directory: {processed_path}")
        shutil.rmtree(processed_path)

    splitted_path = dirname / "../data_set/splitted"
    if splitted_path.exists():
        print(f"Removing old splitted directory: {splitted_path}")
        shutil.rmtree(splitted_path)
    
    print("Loading raw images...")
    all_images = load_images(LoadingType.RAW)
    labels = [img.category for img in all_images]

    # Split data (80% train, 10% val, 10% test)
    # First, separate out the test set (10%)
    train_val_imgs, test_imgs, train_val_labels, _ = train_test_split(
        all_images, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Next, split the remainder into training (80%) and validation (10%)
    # val_size should be 0.1 / 0.9 = 0.1111...
    train_imgs, val_imgs, _, _ = train_test_split(
        train_val_imgs, train_val_labels, test_size=(0.1/0.9), random_state=42, stratify=train_val_labels
    )

    print(f"Total images: {len(all_images)}")
    print(f"Training set size: {len(train_imgs)}")
    print(f"Validation set size: {len(val_imgs)}")
    print(f"Test set size: {len(test_imgs)}")

    # Process and save training data with augmentation
    print("\nProcessing training data with augmentation...")
    train_proc_imgs = read_images(train_imgs)
    train_proc_imgs = change_resolution(train_proc_imgs)
    train_proc_imgs = variate_brightness_contrast(train_proc_imgs)
    train_proc_imgs = variate_rotation(train_proc_imgs)
    train_proc_imgs = variate_flip(train_proc_imgs)
    save_images_to_split(train_proc_imgs, "train")
    print(f"Saved {len(train_proc_imgs)} augmented training images.")

    # Process and save validation data (no augmentation)
    print("\nProcessing validation data...")
    val_proc_imgs = read_images(val_imgs)
    val_proc_imgs = change_resolution(val_proc_imgs)
    save_images_to_split(val_proc_imgs, "val")
    print(f"Saved {len(val_proc_imgs)} validation images.")

    # Process and save test data (no augmentation)
    print("\nProcessing test data...")
    test_proc_imgs = read_images(test_imgs)
    test_proc_imgs = change_resolution(test_proc_imgs)
    save_images_to_split(test_proc_imgs, "test")
    print(f"Saved {len(test_proc_imgs)} test images.")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()

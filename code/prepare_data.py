from dataclasses import dataclass
from cv2.typing import MatLike
from loading_data import Category, Image, load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path
import uuid
import random
import shutil

dirname = Path(__file__).parent

@dataclass
class ProcessingImage:
    path: str
    mat: MatLike
    category: Category

def read_images(images: list[Image]) -> list[ProcessingImage]:
    result: list[ProcessingImage] = []

    for img in images:
        read_img = cv.imread(img.image_path)
        if read_img is None:
            continue

        result.append(ProcessingImage(
            mat=read_img,
            category=img.category,
            path=img.image_path
        ))

    return result

def change_resolution(images: list[ProcessingImage], size=(224, 224)):
    result: list[ProcessingImage] = []

    for img in images:
        resized_img = cv.resize(img.mat, size, interpolation=cv.INTER_AREA)
        result.append(ProcessingImage(
            mat=resized_img,
            category=img.category,
            path=img.path
        ))

    return result

def variate_brightness_contrast(images: list[ProcessingImage]) -> list[ProcessingImage]:
    result: list[ProcessingImage] = []

    for img in images:
        result.append(ProcessingImage(
            mat=img.mat,
            category=img.category,
            path=img.path
        ))

        for _ in range(5):
            alpha = random.uniform(1.0, 1.3)  # Contrast control
            beta = random.randint(0, 30)     # Brightness control

            adjusted = cv.convertScaleAbs(img.mat, alpha=alpha, beta=beta)
            result.append(ProcessingImage(
                mat=adjusted,
                category=img.category,
                path=img.path
            ))

    return result

def variate_rotation(images: list[ProcessingImage]) -> list[ProcessingImage]:
    result: list[ProcessingImage] = []

    for img in images:
        result.append(ProcessingImage(
            mat=img.mat,
            category=img.category,
            path=img.path
        ))

        for angle in [-10, -5, 5, 10]:
            if random.random() < 0.9:
                continue

            (h, w) = img.mat.shape[:2]
            center = (w // 2, h // 2)

            M = cv.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv.warpAffine(img.mat, M, (w, h))

            result.append(ProcessingImage(
                mat=rotated,
                category=img.category,
                path=img.path
            ))

    return result

def variate_flip(images: list[ProcessingImage]) -> list[ProcessingImage]:
    result: list[ProcessingImage] = []

    for img in images:
        result.append(ProcessingImage(
            mat=img.mat,
            category=img.category,
            path=img.path
        ))

        if random.random() < 0.9:
            continue

        flipped = cv.flip(img.mat, 1)  # Horizontal flip
        result.append(ProcessingImage(
            mat=flipped,
            category=img.category,
            path=img.path
        ))

    return result

def save_images(imgs: list[ProcessingImage]) -> None:
    for image in imgs:
        base_name = os.path.basename(image.path).split('.')[0]
        file_name = f"{base_name}{uuid.uuid4()}.jpg"
        category = image.category.value
        folder_path = (dirname / "../data_set/processed" / category).resolve()
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        cv.imwrite(save_path, image.mat)

if __name__ == "__main__":
    if (dirname / "../data_set/processed").exists():
        shutil.rmtree(dirname / "../data_set/processed")

    images = load_images(LoadingType.RAW)
    loaded_imgs = read_images(images)
    loaded_imgs = change_resolution(loaded_imgs, size=(224, 224))
    loaded_imgs = variate_brightness_contrast(loaded_imgs)
    loaded_imgs = variate_rotation(loaded_imgs)
    loaded_imgs = variate_flip(loaded_imgs)

    print(f"Saving {len(loaded_imgs)} images...")
    save_images(loaded_imgs)

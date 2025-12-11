from dataclasses import dataclass
from cv2.typing import MatLike
from loading_data import Category, Image, load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path

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


def save_images(imgs: list[ProcessingImage]) -> None:
    for image in imgs:
        file_name = os.path.basename(image.path)
        category = image.category.value
        folder_path = (dirname / "../data_set/processed" / category).resolve()
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        cv.imwrite(save_path, image.mat)

if __name__ == "__main__":
    images = load_images(LoadingType.RAW)
    loaded_imgs = read_images(images)
    loaded_imgs = change_resolution(loaded_imgs, size=(224, 224))
    save_images(loaded_imgs)

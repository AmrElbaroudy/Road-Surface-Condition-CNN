from loading_data import load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path

dirname = Path(__file__).parent


def changeRes(size=(224, 224)):
    images = load_images(LoadingType.RAW)
    for img in images:
        read_img = cv.imread(img.image_path)
        resized_img = cv.resize(read_img, size, interpolation=cv.INTER_AREA)
        save_image(resized_img, img.category, os.path.basename(img.image_path))


def save_image(img, category: str, file_name: str) -> None:
    folder_path = (
        dirname / "../data_set/processed" / category
    ).resolve()  # * another way for getting absolute path

    os.makedirs(
        folder_path, exist_ok=True
    )  # * make directory if it doesn't exist otherwise photo won't be saved

    save_path = os.path.join(folder_path, file_name)

    cv.imwrite(save_path, img)

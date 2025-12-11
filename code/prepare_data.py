from loading_data import load_images, LoadingType
import cv2 as cv
import os
from pathlib import Path

dirname = Path(__file__).parent

def changeRes(size=(224, 224)):
    images = load_images(LoadingType.RAW)
    for img in images:
        read_img = cv.imread(img.image_path)
        if read_img is None:
            continue
        resized_img = cv.resize(read_img, size, interpolation=cv.INTER_AREA)
        save_image(resized_img, img.category, os.path.basename(img.image_path))


def save_image(img, category: str, file_name: str) -> None:
    folder_path = (dirname / "../data_set/processed" / category).resolve()
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, file_name)
    cv.imwrite(save_path, img)

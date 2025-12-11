import os
from dataclasses import dataclass
from enum import StrEnum

class Category(StrEnum):
    GOOD = "good"
    POOR = "poor"
    SATISFACTORY = "satisfactory"
    VERY_POOR = "very_poor"

@dataclass
class Image:
    image_path: str
    category: Category

class LoadingType(StrEnum):
    RAW = "raw"
    PROCESSED = "processed"

def load_images(type: LoadingType) -> list[Image]:
    result: list[Image] = []

    folder_path = ""
    if type == LoadingType.RAW:
        folder_path = "../data_set/raw" 
    elif type == LoadingType.PROCESSED:
        folder_path = "../data_set/processed"

    print(folder_path)

    for subdir, _, files in os.walk(folder_path):
        if subdir == folder_path or subdir == "":
            continue

        for file in files:
            category = subdir.split(os.sep)[-1]
            result.append(Image(
                image_path=os.path.join(subdir, file),
                category=Category(category)
            ))

    return result

if __name__ == "__main__":
    images = load_images(LoadingType.RAW)
    print(f"Loaded {len(images)} images.")
    for img in images[:5]:
        print(img)

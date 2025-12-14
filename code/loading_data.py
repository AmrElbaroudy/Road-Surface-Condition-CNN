import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

dirname = os.path.dirname(__file__)

class Category(StrEnum):
    GOOD = "good"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class Image:
    image_path: str
    category: Category

class LoadingType(StrEnum):
    RAW = "raw"
    PROCESSED = "processed"
    SPLITTED = "splitted"

def load_images(type: LoadingType, split_name: Optional[str] = None) -> list[Image]:
    result: list[Image] = []

    folder_path = ""
    if type == LoadingType.RAW:
        folder_path = os.path.join(dirname, "../data_set/raw")
    elif type == LoadingType.PROCESSED:
        folder_path = os.path.join(dirname, "../data_set/processed")
    elif type == LoadingType.SPLITTED:
        if split_name is None:
            raise ValueError("split_name must be provided when loading splitted data")
        folder_path = os.path.join(dirname, f"../data_set/splitted/{split_name}")
    
    folder_path = os.path.abspath(folder_path) # remove ../ from path

    for subdir, _, files in os.walk(folder_path):
        if subdir == folder_path or subdir == "":
            continue
        
        category_str = os.path.basename(subdir)

        category: Optional[Category] = None
        if category_str == "satisfactory":
            category = Category.POOR
        elif category_str in list(map(lambda c: c.value, Category)):
             category = Category(category_str)
        else:
            continue

        for file in files:
            result.append(Image(
                image_path=os.path.join(subdir, file),
                category=category
            ))

    return result

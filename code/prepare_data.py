from loading_data import load_images, LoadingType

WIDTH = 224
HEIGHT = 224

def prepare_images():
    images = load_images(LoadingType.RAW)

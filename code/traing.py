from loading_data import LoadingType, load_images

images = load_images(LoadingType.PROCESSED)
print(f"Loaded {len(images)} images for training.")

print("First 5 image paths and categories:")
for img in images[:5]:
    print(f"Path: {img.image_path}, Category: {img.category}")

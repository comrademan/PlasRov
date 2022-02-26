import albumentations as A
import cv2
import glob
import os

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=1),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=1),
    A.MultiplicativeNoise(p=1)
])

for index, name in enumerate(glob.glob('C:/Users/kevin/Documents/Tensorflow/scripts/preprocessing/Train/*.jpg')):
    print(name)
    newpath = 'C:/Users/kevin/Documents/Tensorflow/scripts/preprocessing/Train/noise%s.jpg' % (index,)
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(name)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    cv2.imwrite(newpath, transformed_image)

for index, name in enumerate(glob.glob('C:/Users/kevin/Documents/Tensorflow/scripts/preprocessing/Test/*.jpg')):
    print(name)
    newpath = 'C:/Users/kevin/Documents/Tensorflow/scripts/preprocessing/Test/noise%s.jpg' % (index,)
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(name)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    cv2.imwrite(newpath, transformed_image)

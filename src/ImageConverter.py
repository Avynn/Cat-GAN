from PIL import Image
import numpy as np
import argparse
import os
import glob
import sys
import io

FLAGS = None

imageData = np.zeros((64,64,3))

dogImage = Image.open("../resources/training_set/dogs/dog.1.jpg")
dogImage = dogImage.resize((64,64))
dogPixels = dogImage.load()
dogPixelsToBytes = io.BytesIO()
dogImage.save(dogPixelsToBytes, format="png")

def convertImages(folderPath, name):
    knowDirectories = []
    i = 0
    for (dirPath, dirNames, fileNames) in os.walk(folderPath):
        if (i == 0):
            knowDirectories = dirNames 
        i += 1
    print(knowDirectories)
    print(i)

# imgX = 0
# imgY = 0

# for i in range(0,(64 * 64)):
#     if(imgX > 63):
#         imgX = 0
#         imgY += 1
#     for j in range(0,3):
#         imageData[imgX][imgY][j] = dogPixels[imgX, imgY][j]
#     imgX += 1

convertImages("../resources/training_set/", 1)

print("ran!")
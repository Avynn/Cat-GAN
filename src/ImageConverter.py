from PIL import Image
import numpy as np


imageData = np.zeros((64,64,3))

dogImage = Image.open("../resources/training_set/dogs/dog.1.jpg")
dogImage = dogImage.resize((64,64))
dogPixels = dogImage.load()

imgX = 0
imgY = 0

for i in range(0,(64 * 64)):
    if(imgX > 63):
        imgX = 0
        imgY += 1
    for j in range(0,3):
        imageData[imgX][imgY][j] = dogPixels[imgX, imgY][j]
    imgX += 1
        

print(imageData)
print("done copying!")
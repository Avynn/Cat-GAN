from PIL import Image

imageData = []

dogImage = Image.open("../resources/training_set/dogs/dog.1.jpg")
dogImage = dogImage.resize((64,64))
dogPixels = dogImage.load()

imgX = 0
imgY = 0

for i in range(0,((64 * 64) - 1)):
    print((imgX, imgY))
    if(imgX > 63):
        imgX = 0
        imgY += 1
    else:
        imageData.append(dogPixels[imgX, imgY])
        imgX += 1

print("done copying!")
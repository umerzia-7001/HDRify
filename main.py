import cv2
import numpy as np
def readImagesAndTimes():

    # list of exposure times
    times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)

    #list of image filenames
    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
    images=[]
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images,times


if __name__ == '__main__':
    # Read images and exposure times
    print("Reading images ... ")

    images, times = readImagesAndTimes()

    # Align input images
    print("Aligning images ... ")
# aligning images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images,images)

# Obtain Camera Response Function (CRF)

calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

#Merge images into an HDR linear image
mergeDb=cv2.createMergeDebevec()
hdrDb=mergeDb.process(images,times,responseDebevec)
# save HDR image
cv2.imwrite("hdrDebevec.hdr",hdrDb)

# Tonemap using Drago's method to obtain 24-bit color image
# using to create 24 pixel bit images from merged images
tonemapDrago= cv2.createTonemapDrago(1.0,0.7)
ldrDrago = tonemapDrago.process(hdrDb)
ldrDrago = 3 * ldrDrago
cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)

# Tonemap using Durand method to obtain 24-bit color image
tonemapDurand = cv2.createTonemapDrago(1.5,4,1.0)
ldrDurand = tonemapDurand.process(hdrDb)
ldrDurand = 3* ldrDurand
cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)

# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(hdrDb)
cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)

# Tonemap using Mantiuk's method to obtain 24-bit color image
tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDb)
ldrMantiuk = 3 * ldrMantiuk
cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)


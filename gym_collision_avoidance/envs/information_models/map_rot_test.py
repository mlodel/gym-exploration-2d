import cv2
import numpy as np
import time
start_time = time.time()

def rotate_image(image, angle, point):

  rot_mat = cv2.getRotationMatrix2D(point, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == '__main__':

    A = np.zeros((20,20), dtype=np.float32)
    # np.fill_diagonal(A,0.5)
    # A[30:35,:] = np.ones((5,40))
    # A[:10, :] = np.ones((10, 40))
    # A[10:15, 0:5] = np.ones((5,5))
    #
    # A[0:5, 0:5] = np.zeros((5, 5))


    pos = (17,17)
    angle = 180
    A[17,17] = 1

    start_time = time.time()

    rotateImage = A

    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]

    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight // 2, imgWidth // 2

    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), 45, 1.0)

    newImageWidth = 30
    newImageHeight = 30

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth / 2) - centreX
    rotationMatrix[1][2] += (newImageHeight / 2) - centreY

    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight), borderValue=1.0)

    map = np.ones((90,90), dtype=np.float32)

    map[30:60,30:60] = rotatingimage

    transform_point = cv2.transform(np.asarray(pos).reshape((1,1,2)), rotationMatrix)[0][0]

    point = transform_point + np.array([30,30])


    rot_mat = cv2.getRotationMatrix2D(tuple(point), 45-angle, 1.0)
    rot2 = cv2.warpAffine(map, rot_mat, map.shape[1::-1], borderValue=1.0)


    final = rot2[point[1]-30:point[1]+30, point[0]-30:point[0]+30]
    final_mirror = cv2.flip(final,1)

    showimage = cv2.resize(final_mirror, (600,600))
    showimage = np.array(showimage*255, dtype=np.uint8)
    cv2.imshow("image", showimage)
    cv2.waitKey(0)

    print("--- %s seconds ---" % (time.time() - start_time))
    a=1
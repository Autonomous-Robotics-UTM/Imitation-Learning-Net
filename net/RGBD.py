import cv2
import numpy as np
def read(path):
    img = cv2.imread(path)
    print(img.shape)
    extra = np.zeros((469,626,1))*100
    #extra = np.zeros((480,640,1))*100
    print(extra.shape)
    result = np.dstack((img, extra))
    print(result.shape)
    new_path = "new.png"

    cv2.imwrite(new_path,result)


    img = cv2.imread(new_path)
    print(img.shape)
    #cv2.imshow("image",result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#read("000046.jpg")
read("image.png")
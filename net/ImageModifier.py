import PIL
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

count = 0
old_labels = None
new_labels = None

def manipulate(path,modified_path,func,num=0):
    global count
    global old_labels
    global new_labels
    old_labels = open(path+"data.csv", "r")
    new_labels = open(modified_path+"data.csv", "w")
    try:
        os.mkdir(modified_path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    for filename in sorted(os.listdir(path)):
        print(count)
        if filename.endswith("jpg"):
            img = cv2.imread(path + filename, cv2.IMREAD_UNCHANGED)
            img,img_path = func(img,modified_path,filename)
            if img_path != None:
                cv2.imwrite(img_path,img)
        if num != 0 and count == num:
            break
        count += 1
    old_labels.close()
    new_labels.close()

def name(index, image):
    name = "0"*(6-len(str(index)))+str(index)
    return name


def crop(image,modified_path,filename):
    global old_labels
    global new_labels
    y = 230
    x = 0
    w = 640
    h = 170
    crop_img = image[y:y+h, x:x+w]
    line = old_labels.readline()
    new_labels.write(line)
    return crop_img,modified_path+filename

def scale(image,modified_path,filename):
    global old_labels
    global new_labels
    dim = (640,360)
    scaled_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    line = old_labels.readline()
    new_labels.write(line)
    return scaled_img,modified_path+filename

def select(image,modified_path,filename):
    global count
    global old_labels
    global new_labels
    selection = 10
    if count%selection == 0:
        new_name = name(count//selection, image)
        line = old_labels.readline()
        new_labels.write(line)
        return image,modified_path+new_name+".jpg"
    return None,None




data_path = os.path.abspath(os.getcwd())+"/data/cropped/"
new_path = os.path.abspath(os.getcwd())+"/data/selected/"
manipulate(data_path,new_path,select)

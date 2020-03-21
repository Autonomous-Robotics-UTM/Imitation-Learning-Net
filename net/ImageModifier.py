import PIL
from PIL import Image
import numpy as np
import math
import cv2
import os

count = 0
old_labels = None
new_labels = None
num_lines = 0

def manipulate(path,modified_path,func,num=0):
    global count
    global num_lines
    for filename in sorted(os.listdir(path)):
        if not filename.endswith("csv"):
            filename = filename[:-4]
            img = read(path + filename)
            img,img_path = func(img,modified_path,filename)
            if img_path != None:
                save(img_path,img)
        if num != 0 and count == num:
            break
        count += 1
        if count%100 == 0:
            print(count)

def name(index):
    name = "0"*(6-len(str(index)))+str(index)
    return name

def copy(image,modified_path,filename):
    line = old_labels.readline()
    new_labels.write(line)
    return image,modified_path+filename

def begin_writing():
    global old_labels
    global new_labels
    old_labels = open(data_path+"data.csv", "r")
    new_labels = open(new_path+"data.csv", "w")
    columns_titles = old_labels.readline()
    new_labels.write(columns_titles)

def finish_writing():
    global old_labels
    global new_labels
    old_labels.close()
    new_labels.close()

def continue_writing():
    global old_labels
    global new_labels
    old_labels = open(data_path+"data.csv", "r")
    new_labels = open(new_path+"data.csv", "a")
    columns_titles = old_labels.readline()

def crop(image,modified_path,filename):
    global old_labels
    global new_labels
    y = 160
    x = 140
    w = 400
    h = 220
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
        index = name(count//selection)
        filename = index+".jpg"
        line = old_labels.readline()
        new_labels.write(line)
        return image,modified_path+filename
    return None,None

def reflect(image,modified_path,filename):
    offset = num_lines-1
    filename = name(int(filename)+offset)
    image = cv2.flip(image,1)
    line = old_labels.readline()
    count,label = line.split(",")
    if float(label) == 0:
        new_label = str(0)
    else:
        new_label = str(-float(label))
    new_line = str(int(count)+offset)+","+new_label+"\n"
    new_labels.write(new_line)
    return image,modified_path+filename

def resize(image,modified_path,filename):
    dim = (200, 100)
    image = cv2.resize( image, dim, interpolation = cv2.INTER_AREA)
    line = old_labels.readline()
    new_labels.write(line)
    return image,modified_path+filename

def xysv(image,modified_path,filename):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = image[:,:,0]/float(255)*(2*np.pi)
    SV = 2*(image[:,:,1:]/float(255))-1
    X = np.sin(H)
    Y = np.cos(H)

    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)
    XYSV = np.dstack((X,Y,SV))
    line = old_labels.readline()
    new_labels.write(line)
    return XYSV,modified_path+filename

def read(path):
    img = cv2.imread(path+".jpg",cv2.IMREAD_UNCHANGED)
    #img = np.load(path+".npy")
    return img

def save(path,image):
    #cv2.imwrite(path+".jpg",image)
    np.save(path, image)


data_path = os.path.abspath(os.getcwd())+"/data/cropped_vflip/"
new_path = os.path.abspath(os.getcwd())+"/data/xysv/"
try:
    os.mkdir(new_path)
except OSError:
    print ("Directory %s already exists" % new_path)

num_lines = sum(1 for line in open(data_path+"data.csv", "r"))

begin_writing()
manipulate(data_path,new_path,xysv)
#continue_writing()
#manipulate(data_path,new_path,reflect)
finish_writing()

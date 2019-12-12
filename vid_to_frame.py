
# Program To Read video
# and Extract Frames
import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2
import os
from PIL import Image
import numpy as np


path   = "data/Video/"
# Path to video file
vid    = "1.mov"
folder = "frames2/"

vidObj = cv2.VideoCapture(path + vid)
# get 3 of 4 video dimensions
success, img = vidObj.read()
height, width, depth = img.shape
dtmax = np.iinfo(img.dtype).max # Get the information of the incoming image type
# count starts at one cuz we just read one
count = 1
# checks whether frames were extracted
success = 1
while success:
    # vidObj object will extract frames
    success, img = vidObj.read()
    if not success:
        break
    # just get dimension
    count += 1


sfactor = 10
#vid_data = np.zeros((count, int(height/sfactor), int(width/sfactor), depth), dtype = 'uint8')
vid_data = np.zeros((count, int(height/sfactor), int(width/sfactor), depth), dtype = 'float16')
dim = (int(width/sfactor), int(height/sfactor))

# restart movie reader
vidObj = cv2.VideoCapture(path + vid)
success = 1
index = 0
data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
while success:
    success, img = vidObj.read()
    if not success:
        break
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    vid_data[index] = img/dtmax
    index += 1

in_hw = np.array(vid_data.shape[1:3])
out_hw = [64, 128]
clip = in_hw - out_hw
clipx = clip // 2
clipy = clipx + out_hw
rem = clip - 2*clipx
clipx = clipx + rem
vid_data = vid_data[:, clipx[0]:clipy[0], clipx[1]:clipy[1], : ]

np.save("data/Video/" + "video1.npy", vid_data)


# pil test image prints
pil_im = Image.fromarray((vid_data[8133]*dtmax).astype('uint8'))
pil_im = Image.fromarray(img)
pil_im.save("resize_test.jpg", "JPEG")





   ##########################
   #         TRASH!         #
   #   THIS I WILL DELETE   #
   ##########################



success, img = vidObj.read()
img.shape
img = cv2.resize(img,(int(width/sfactor), int(height/sfactor)))
img.shape
 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(img)
pix = np.array(pil_im, dtype='float16')
pix.shape
pil_im.save("resize_test.jpg", "JPEG")
#pil_im_bw.save("desat_test.jpg", "JPEG")

cv2.getD    (img)
img = cv2.resize(img, (72,128)
img = img_to_array(img)
while success:
    # vidObj object calls read
    # function extract frames
    success, img = vidObj.read()
    img = img_to_array(img)
    # Saves the frames with frame-count
    #cv2.imwrite(path + folder + "f%d.jpg" % count, image)
    count += 1

FrameCapture("data/Video/", "2.mov", "frames2/")

# Function to extract frames
def FrameCapture(path, vid, folder):
    # Path to video file
    vidObj = cv2.VideoCapture(path + vid)
    # Used as counter variable
    try:
        os.stat(path + folder)
    except:
        os.mkdir(path + folder)
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        # vidObj object calls read
        # function extract frames
        success, img = vidObj.read()
        img = img_to_array(img)
        # Saves the frames with frame-count
        #cv2.imwrite(path + folder + "f%d.jpg" % count, image)
        count += 1


   ##########################
   #  DATA PROCESSING TIME  #
   #   THIS I WILL DELETE   #
   ##########################


img_data_path = "data/Video/frames1/"
# load the image
img = load_img(img_data_path + "f1.jpg")
# convert to numpy array
img = img_to_array(img)
# save shape to pass to convolution
img_shape = img.shape
# convert back to image
img_pil = array_to_img(img_array)
print(img_shape)

   ##########################
   #  DATA PROCESSING TIME  #
   ##########################


img_data_path = "data/Video/frames1/"
count = 0
for im in os.listdir(img_data_path):
    count += 1

count = count - 1
# load the image
img = load_img(img_data_path + "f1.jpg")
# convert to numpy array
img = img_to_array(img)
# save shape to pass to convolution
img_shape = list(img.shape)
# save shape to pass to convolution
img_shape.reverse()
img_shape.append(count)
img_shape.reverse()
img_shape
data = np.zeros(img_shape)
data.shape

for i, im in enumerate(list(range(count))):
    # load the image
    img = load_img(img_data_path + "f{}.jpg".format(im))
    # convert to numpy array
    img = img_to_array(img)
    #print(img.shape)
    data[i,:,:,:] = img

# Additional reshape
# Load in not processed
vdata = np.load("data/Video/video1.npy")
vdata.shape
adata = np.load("data/Video/video1.npy")
adata.shape
1280-720
560/2

data_newshape = np.pad(data, ((0,0),(280,280),(0,0),(0,0)), mode='constant')
data_newshape[3,279,:,1]
data_newshape.shape
datasmall = data_newshape.astype('float16')
np.save("data/Video/video1rs.npy", datasmall)

datasmall = data.astype('float16')
np.save("data/Video/" + "video1.npy", datasmall)
data2 = np.load(img_data_path + "video1.npy")
data2[8128,:,:,:]
data[50,:,:,:]



# trash # zone # trash # zone # trash # zone # trash # zone # trash # zone # trash # zone # trash # zone

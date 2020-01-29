import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image

def make_video(training_vid_path, training_audio_path, name, generator_path, length):
    # Get the information of the incoming image type
    dtmax = np.iinfo('uint8').max
    generator = load_model(generator_path)
    print("Loading training data.")
    training_vid = np.load(training_vid_path)
    training_aud = np.load(training_audio_path)
    #
    # First lets make a reference video
    #
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    out = cv2.VideoWriter("original.avi", fourcc, fps, (training_vid.shape[2], training_vid.shape[1]))
    for f in training_vid:
        out.write( (f*dtmax).astype('uint8') )
    # close out the video writer
    print("video one written")
    out.release()
    first_vid = np.zeros(training_vid.shape, dtype='float16')
    for i in range(first_vid.shape[0]):
        first_vid[i,:,:,:] = training_vid[0,:,:,:]

    vid = generator.predict([first_vid[:length], training_aud[:length]])

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    out = cv2.VideoWriter(name, fourcc, fps, (vid.shape[2], vid.shape[1]))
    for f in vid:
        out.write( (f*dtmax).astype('uint8') )
    # close out the video writer
    out.release()
    print("video two written")

# Because of time needed, we use a Numpy preprocessed file.
training_vid_path = "/home/dl-group/data/Video/video1.npy"
training_audio_path = "/home/dl-group/data/Audio/audio1.npy"
name = "test-e180.avi"
generator_path = './generator-e180.h5'
length = 1000

make_video(training_vid_path, training_audio_path, name, generator_path, length)







import os
import pandas as pd
import numpy as np
# audio editing libs
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt

sound = AudioSegment.from_file('1.wav')
#frame_file_path = "frames1/"
#np_frame_file_path = "npframes1/"
#frame_file_path = "frames1-half/"
#np_frame_file_path = "npframes1-half/"
frame_file_path = "frames1-forth/"
np_frame_file_path = "npframes1-forth/"
#num_cuts = 8135
#num_cuts = 8135 // 2
num_cuts = 8135 // 4
n_mels = 320
n_fft = 2048
hop_length = 100
try:
    os.stat(frame_file_path)
except:
    os.mkdir(frame_file_path)

try:
    os.stat(np_frame_file_path)
except:
    os.mkdir(np_frame_file_path)

# the number of cuts/frames it should match the video frames.
size_frame = len(sound) // num_cuts
step_size = len(sound) / num_cuts
sound_set = []
center = size_frame
center_true = step_size
for i in range(num_cuts):
    start = center - size_frame
    stop  = center + size_frame
    # sanity check
    if start < 0:
        start = 0
    if stop > len(sound):
        stop = len(sound)
    sound_set.append(sound[start:stop])
    center_true = center_true + step_size
    center = int(center_true)


f_num = []
for i, frame in enumerate(sound_set):
    f_num.append(i)
    frame.export(frame_file_path + "{}.wav".format(i),format="wav")

for i in f_num:
    wav = "{}.wav".format(i)
    # here kaiser_fast is a technique used for faster extraction
    audio, sample_rate = librosa.load(frame_file_path+wav, res_type='kaiser_fast')
    #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels)
    #mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40
    #break
    ## we extract mfcc feature from data
    ##mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #feature = mfccs
    #print(feature)
    ##print(label)
    S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_fft=n_fft,



   ##########################
   #  DATA PROCESSING TIME  #
   ##########################

import numpy as np
import os


# load the image
img = np.load("data/Audio/npframes1/1.npy")
count = 0
for im in os.listdir("data/Audio/npframes1/"):
    count += 1

count = count - 1
# convert to numpy array
img = img_to_array(img)
# save shape to pass to convolution
img_shape = list(img.shape)
img_shape.reverse()
img_shape.append(count)
img_shape.reverse()
img_shape
data = np.zeros(img_shape)
data.shape

for i, im in enumerate(list(range(count))):
    img = np.load("data/Audio/npframes1/{}.npy".format(im))
    if img.shape == (320,8):
        #print(img.shape)
        img = np.reshape(img, (320, 8,1))
        data[i,:,:,:] = img
    else:
        print("skiped ", im)

np.save("data/Audio/audio1.npy", data)
data2 = np.load("data/Audio/audio1.npy")
data2[8128,:,:,:]
data[50,:,:,:]







# trash # zone # trash # zone # trash # zone # trash # zone # trash # zone # trash # zone
                                       hop_length=hop_length,

                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_nb = np.asmatrix(S_DB)
    np.save(np_frame_file_path + "{}".format(i), S_nb)

print(mel_spec)
print(mel_db)

        librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length,
                                 x_axis='time', y_axis='mel');
        plt.colorbar(format='%+2.0f dB');
        plt.show()


sound_set[0].get_array_of_samples()
temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']

def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')
   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None
   feature = mfccs
   label = row.Class
   return [feature, label]


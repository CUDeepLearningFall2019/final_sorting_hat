import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# load all the layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
# load model
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model, Model
# load back end
from tensorflow.keras import backend
# load optimizers
from tensorflow.keras.optimizers import Adam
# load other tensor stuff
import tensorflow as tf
# load other stuff stuff
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# grph = tf.get_default_graph()
# sess = tf.Session(graph=grph)
#
# # Generation resolution - Must be square
# # Training data is also scaled to this.
# # Note GENERATEoRES higher than 4 will blow Google CoLab's memory.
# GENERATE_RES = 2 # (1=32, 2=64, 3=96, etc.)
# GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
# HEIGHT = 720
# WIDTH = 1280
# IMAGE_CHANNELS = 3
# INPUT_SHAPE = (HEIGHT, WIDTH)
#
# # Preview image
# PREVIEW_ROWS = 4
# PREVIEW_COLS = 7
# PREVIEW_MARGIN = 16
# SAVE_FREQ = 100
#
# # Size vector to generate images from
# SEED_SIZE = 10
#
# # Configuration
# DATA_PATH = ''
# EPOCHS = 10
# BATCH_SIZE = 64
#
# #print(f"Will generate {GENERATE_SQUARE}px square images.")
#
# # Image set has 11,682 images.  Can take over an hour for initial preprocessing.
# # Because of this time needed, save a Numpy preprocessed file.
# # Note, that file is large enough to cause problems for some versions of Pickle,
# # so Numpy binary files are used.
# training_vid_path = "/home/dl-group/data/Video/video1.npy"
# training_audio_path = "/home/dl-group/data/Audio/audio1.npy"
#
# print("Loading data pickle...")
# training_vid = np.load(training_vid_path)
# training_aud = np.load(training_audio_path)
#
# # take the first video and make a matrix used as part of generator context
# first_vid = np.zeros(training_vid.shape, dtype='float16')
# for i in range(first_vid.shape[0]):
#     first_vid[i,:,:,:] = training_vid[0,:,:,:]
#
# training_vid.shape
# training_aud.shape

def noise_enc():
    ne = None
    return ne

###Rory code

def context_enc():
    # one aproch
    # https://towardsdatascience.com/an-approach-towards-convolutional-recurrent-neural-networks-f54cbeecd4a6

    # vary setings
    shape_in = int(320), int(8), int(1)
    shape_out = int(8134), int(120)
    dropoutrate = 0.3

    x_start = Input(shape=(shape_in))
    x = x_start

    for _i, _cnt in enumerate((2, 2)):
        x = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(x)
        x = BatchNormalization(axis=-1)(x)
        #x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropoutrate)(x)

    x = Permute((2, 1, 3))(x)
    x = Reshape((1, 16000))(x)

    # The Gru/recurrent portion
    # Get some knowledge
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    for r in (10,10):
        x = Bidirectional(
                GRU(r,
                    activation='tanh',
                    dropout=dropoutrate,
                    recurrent_dropout=dropoutrate,
                    return_sequences=True),
                merge_mode='concat')(x)
        for f in ((2,2)):
            x = TimeDistributed(Dense(f))(x)

    x = Dropout(dropoutrate)(x)
    x = TimeDistributed(Dense(880))(x)
    # arbitrary reshape may be a problem
    x = Reshape((22,40,1))(x)
    out = Activation('sigmoid', name='strong_out')(x)
    #audio_context = Model(inputs=x_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()
    ce = out
    #ce = audio_context
    return ce, x_start

# Q0 can I flatten and Reshape and still get good results
# Q1 if I do that were should my GRU layers go
# Q3 how do I ensure that the down block matches the shape of the up block
# Q4

#UNET Functions

def down_block(x, filters, kernal_size = (3, 3), padding ='same', strides=1):
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(x)
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernal_size =(3, 3), padding='same', strides= 1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(concat)
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(c)
    return c

def up_block1(x, skip, filters, kernal_size =(3, 3), padding='same', strides= 1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([x, skip])
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(concat)
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(c)
    return c

def bottleneck(IE, AE, NE, filters, kernal_size = (3, 3), padding ='same', strides=1):
    #join_shape = (22,40, 128)
    join_shape = (4, 8, 128)

    IE = Flatten()(IE)
    AE = Flatten()(AE)
    concat1 = Concatenate()([IE, AE])
    concat1 = Dense(np.prod(join_shape))(concat1)
    concat1 = Reshape(join_shape)(concat1)


    #concat1 = Concatenate()([p5, context_enc()])
    #filters = f[5]
    #because we are not using gausian noise, I have commented out that concatentation and only passed the first concat
    #concat2 = Concatenate()([concat1, NE])
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(concat1)
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(c)
    #c = Activation('linear')(c)
    #Flatten()(c)
    #c = Dense((45, 80, 63))(c)
    # fix this
    # c = Reshape((45, 80, 63))(c)
    #c = concat1
    return c

#def Gener(input_dim, image_channels):
#def build_generator(discrim):
def build_generator():
    f= [8, 16, 32, 64, 128, 256]
    #Filters per layers

    #Data shape entering the convolusion
    inputs = Input((64,128,3))


    #Input layer
    p0= inputs

    #Down bloc encoding
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
    c5, p5 = down_block(p4, f[4]) # output is 4,8, 128
    # test = Model(inputs=inputs, outputs=p5)
    # test.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    # test.summary()

    #switch over layer to up bloc decoder
    ne = noise_enc()
    ae, inputs2 = context_enc()

    bn = bottleneck(p5, ae, ne, f[5])

    #Up bloc decoding
    u1 = up_block1(bn, c5, f[4])
    u2 = up_block(u1, c4, f[3])
    u3 = up_block(u2, c3, f[2])
    u4 = up_block(u3, c2, f[1])
    u5 = up_block(u4, c1, f[0])

    #autoencoder egress layer. Flatten and any perceptron layers would succeed this layer
    outputs = Conv2D(3, (1, 1), padding='same', activation = 'tanh')(u5)
    #outputs = Conv2D(3, (1, 1), padding='same', activation = 'tanh')(bn)

    #Keras model output
    model = Model([inputs, inputs2], outputs, name='gener')
    #model.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model, [inputs, inputs2]



def build_discriminator():
    # one aproch
    # https://towardsdatascience.com/an-approach-towards-convolutional-recurrent-neural-networks-f54cbeecd4a6

    # vary setings
    shape_in_x = int(320), int(8), int(1)
    shape_in_y = int(64), int(128), int(3)
    #shape_out = int(8134), int(120)
    dropoutrate = 0.3

    x_start = Input(shape=(shape_in_x))
    x = x_start

    for _i, _cnt in enumerate((2, 2)):
        x = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(x)
        x = BatchNormalization(axis=-1)(x)
        #x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropoutrate)(x)

    x = Flatten()(x) # shape out = none 16000
    #x = Permute((2, 1, 3))(x)
    #x = Reshape((1, 16000))(x)

    #out = Activation('sigmoid', name='strong_out')(x)
    #audio_context = Model(inputs=x_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()

    y_start = Input(shape=(shape_in_y))
    y = y_start

    for _i, _cnt in enumerate((2, 2)):
        y = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(y)
        y = BatchNormalization(axis=-1)(y)
        #y = Activation('relu')(y)
        y = LeakyReLU()(y)
        #y = MayPooling2D(pool_size=(2,2), dim_ordering="th" )(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(dropoutrate)(y)

    y = Flatten()(y)
    #y = Permute((2, 1, 3))(y)
    #y = Reshape((1, 5760000))(y)
    #out = Activation('sigmoid', name='strong_out')(y)
    #audio_context = Model(inputs=y_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()# (None, 51200)

    z = Concatenate()([y, x])
    #z = Reshape((-1, 1))(z) # Why dose this not work?
    z = Reshape((67200, 1))(z)
    #out = Activation('sigmoid', name='strong_out')(z)
    #audio_context = Model(inputs=[x_start,y_start], outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary() # (None, 67200)

    # The Gru/recurrent portion
    # Get some knowledge
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    for r in (10,10):
        z = Bidirectional(
                GRU(r,
                    activation='tanh',
                    dropout=dropoutrate,
                    recurrent_dropout=dropoutrate,
                    return_sequences=True),
                merge_mode='concat')(z)
        for f in ((2,2)):
            z = TimeDistributed(Dense(f))(z)

    #z = Dropout(dropoutrate)(z)
    #z = TimeDistributed(Dense(880))(z)
    # arbitrary reshape may be a problem
    z = Flatten()(z)
    z = Dropout(dropoutrate)(z)
    z = Dense(1,activation = 'sigmoid')(z)

    #out = Activation('sigmoid', name='strong_out')(z)
    descrim = Model(inputs=[y_start, x_start], outputs=z)
    descrim.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #descrim.summary()
    #ce = audio_context
    return descrim, x_start



discriminator, aud_input = build_discriminator()
#discriminator.summary()
generator, inputs = build_generator()
gd_joint = generator(inputs)
#Keras model output
#model = Model([inputs, inputs2], outputs, name='gener')
#model.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator([gd_joint, aud_input])

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
inputs.append(aud_input)
combination = Model(inputs, validity)
combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# Image set has 11,682 images.  Can take over an hour for initial preprocessing.
# Because of this time needed, save a Numpy preprocessed file.
# Note, that file is large enough to cause problems for sume verisons of Pickle,
# so Numpy binary files are used.
training_vid_path = "/home/dl-group/data/Video/video1.npy"
training_audio_path = "/home/dl-group/data/Audio/audio1.npy"

print("Loading training data.")
training_vid = np.load(training_vid_path)
training_aud = np.load(training_audio_path)
#training_vid = np.load(mini_vid_path)
#training_aud = np.load(mini_audio_path)
# take the first video and make a matrix used as part of generator context
first_vid = np.zeros(training_vid.shape, dtype='float16')
for i in range(first_vid.shape[0]):
    first_vid[i,:,:,:] = training_vid[0,:,:,:]

training_vid.shape
training_aud.shape

#
# Training block
#

#iter = 10
#BATCH_SIZE = 2
#save_dir = "./gener_output"
#start = 0
#for step in range(iter):
#rand_lat = np.random.normal(size=(BATCH_SIZE, 64))
#generated_images = generator.predict([first_vid[:10],training_aud[:10]])
#stop = start+BATCH_SIZE
#real_image = [training_vid[start:stop], training_aud[start:stop]]
#0combined_image =
#
#
#
##random_input = Input(shape=(SEED_SIZE,))
#
#generated_image = gener()
#
#validity = discriminator(generated_image)
#combined = Model(random_input,validity)
#combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
#
#y_real = np.ones((64, BATCH_SIZE, BATCH_SIZE, 3))
#y_real = y_real.reshape((-1, 64, 64, 3))
#
#y_fake = np.zeros((BATCH_SIZE,1))
#
#fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

cnt = 1
num_epoch = 10
batch_size = 10
seed_size = 42
work_path = './'
save_freq = num_epoch - 1

for epoch in range(num_epoch):
    idi = np.random.randint(0, training_vid.shape[0]-batch_size)
    idx = list(range(idi, idi + batch_size))
    x_real_vid = training_vid[idx]
    x_real_aud = training_aud[idx]
    image_context = first_vid[idx]
    # Generate some images
    # seed = np.random.normal(0,1,(batch_size,seed_size))
    x_fake_vid = generator.predict([image_context, x_real_aud])
    y_real = np.ones((batch_size))
    y_fake = np.zeros((batch_size))
    print("xxxx")
    print(x_real.shape)
    print(y_real.shape)
    # Train discriminator on real and fake
    discriminator_metric_real = discriminator.train_on_batch([x_real_vid, x_real_aud], y_real)
    discriminator_metric_generated = discriminator.train_on_batch([x_fake_vid, x_real_aud], y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)
    # Train generator on Calculate losses
    # y_pred = discriminator.predict([x_real_aud, x_fake_vid])
    generator_metric = combination.train_on_batch([image_context, x_real_aud, x_real_aud], y_pred)
    # Time for an update?
    if epoch % save_freq == 0:
        save_images(cnt, fixed_seed)
        cnt += 1
        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
    generator.save(os.path.join(work_path,"face_generator.h5"))


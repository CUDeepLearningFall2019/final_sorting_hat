from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, LeakyReLU, MaxPool2D, Input
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from PIL import Image

grph = tf.get_default_graph()
sess = tf.Session(graph=grph)

from resizeimage import resizeimage

#try:
    #from google.colab import drive
    #drive.mount('/content/drive', force_remount=True)
    #COLAB = True
    #print("Note: using Google CoLab")
    #%tensorflow_version 2.x
#except:
    #print("Note: not using Google CoLab")
    #COLAB = True

# Generation resolution - Must be square
# Training data is also scaled to this.
# Note GENERATE_RES higher than 4 will blow Google CoLab's memory.
GENERATE_RES = 2 # (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16
SAVE_FREQ = 100

# Size vector to generate images from
SEED_SIZE = 10

# Configuration
DATA_PATH = ''
EPOCHS = 10
BATCH_SIZE = 64

print(f"Will generate {GENERATE_SQUARE}px square images.")

# Image set has 11,682 images.  Can take over an hour for initial preprocessing.
# Because of this time needed, save a Numpy preprocessed file.
# Note, that file is large enough to cause problems for sume verisons of Pickle,
# so Numpy binary files are used.
training_binary_path = os.path.join(DATA_PATH,f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
  print("Loading training images...")

  training_data = []
  faces_path = os.path.join(DATA_PATH,'frames_test')
  for filename in tqdm(os.listdir(faces_path)):
      path = os.path.join(faces_path,filename)
      image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)
      training_data.append(np.asarray(image))
  training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))
  training_data = training_data / 127.5 - 1.

  print("Saving training image binary...")
  np.save(training_binary_path,training_data)
else:
  print("Loading previous training pickle...")
  training_data = np.load(training_binary_path)


'''
def build_generator(seed_size, channels):

	model = Sequential()

	model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
	model.add(Reshape((4,4,256)))

	model.add(UpSampling2D())
	model.add(Conv2D(256,kernel_size=3,padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))

	model.add(UpSampling2D())
	model.add(Conv2D(256,kernel_size=3,padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))

	# Output resolution, additional upsampling
	for i in range(GENERATE_RES):
		model.add(UpSampling2D())
		model.add(Conv2D(128,kernel_size=3,padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))

	# Final CNN layer
	model.add(Conv2D(channels,kernel_size=3,padding="same"))
	model.add(Activation("tanh"))

	input = Input(shape=(seed_size,))
	generated_image = model(input)

	return Model(input,generated_image, name = "Generator")

'''

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
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        spec_x = Dropout(dropoutrate)(x)
    x = Permute((2, 1, 3))(x)
    x = Reshape((1, -1))(x)
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
    x = TimeDistributed(Dense(shape_out[-1]))(x)
    out = Activation('sigmoid', name='strong_out')(x)
    audio_context = Model(inputs=x_start, outputs=out)
    audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    ce = audio_context
    return ce

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

def bottleneck(IE, AE, NE, filters, kernal_size = (3, 3), padding ='same', strides=1):
    concat1 = Concatenate()([IE, AE])
    #because we are not using gausian noise, I have commented out that concatentation and only passed the first concat
    #concat2 = Concatenate()([concat1, NE])
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(concat1)
    c= Conv2D(filters, kernal_size, padding=padding, strides=strides, activation='relu')(c)
    return c



def Gener():
    #Filters per layers
    f= [8, 16, 32, 64, 128, 256]

    #Data shape entering the convolusion
    inputs = Input((image_size, image_size, 3))


    #Input layer
    p0= inputs

    #Down bloc encoding
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
    c5, p5 = down_block(p4, f[4])

    #switch over layer to up bloc decoder
    ne = noise_enc()
    ae = context_enc()


    bn = bottleneck(p5, ae, ne, f[5])

    #Up bloc decoding
    u1 = up_block(bn, c5, f[4])
    u2 = up_block(u1, c4, f[3])
    u3 = up_block(u2, c3, f[2])
    u4 = up_block(u3, c2, f[1])
    u5 = up_block(u4, c1, f[0])

    #autoencoder egress layer. Flatten and any perceptron layers would succeed this layer
    outputs = Conv2D(3, (1, 1), padding='same', activation = 'tanh')(u5)


    #Keras model output
    model = Model(inputs, outputs)

    return model



def build_discriminator(image_shape):

#first layer is concatenated image

	model = Sequential()
	model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(ZeroPadding2D(padding=((0,1),(0,1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.add(Reshape((64, 64, 3)))
	input_image = Input(shape=image_shape)

	validity = model(input_image)

	return Model(input_image, validity, name = "Discriminator")

def save_images(cnt,noise):
  image_array = np.full((
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)),
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3),
      255, dtype=np.uint8)

  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
        image_count += 1


  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)
optimizer = Adam(1.5e-4,0.5) # learning rate and momentum adjusted from paper

discriminator = build_discriminator(image_shape)
discriminator.trainable = False
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
generator = build_generator(SEED_SIZE,IMAGE_CHANNELS)

random_input = Input(shape=(SEED_SIZE,))

generated_image = generator(random_input)

validity = discriminator(generated_image)
combined = Model(random_input,validity)
combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

y_real = np.ones((64, BATCH_SIZE, BATCH_SIZE, 3))
y_real = y_real.reshape((-1, 64, 64, 3))

y_fake = np.zeros((BATCH_SIZE,1))

fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

cnt = 1
for epoch in range(EPOCHS):
    idx = np.random.randint(0,training_data.shape[0],BATCH_SIZE)
    x_real = training_data[idx]
    x_real = x_real.reshape((-1, 64, 64, 3))

    # Generate some images
    seed = np.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))
    x_fake = generator.predict(seed)

    print("xxxx")
    print(x_real.shape)
    print(y_real.shape)
    # Train discriminator on real and fake

    discriminator_metric_real = discriminator.train_on_batch(x_real,y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)

    # Train generator on Calculate losses
    generator_metric = combined.train_on_batch(seed,y_real)

    # Time for an update?
    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_seed)
        cnt += 1
        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

generator.save(os.path.join(DATA_PATH,"face_generator.h5"))


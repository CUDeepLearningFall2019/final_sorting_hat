from tensorflow.keras.models import  load_model, Model
# load optimizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from model_build import build_discriminator, build_generator
# this checks the tf we are using
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

discriminator, aud_input = build_discriminator()
#discriminator = load_model('./discriminator-e30.h5')
#discriminator.summary()
generator, inputs = build_generator()
#generator = load_model('./generator-e30.h5')
gd_joint = generator(inputs)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator([gd_joint, aud_input])

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
inputs.append(aud_input)
combination = Model(inputs, validity)
combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# Because of time needed, we use a Numpy preprocessed file.
training_vid_path = "/home/dl-group/data/Video/video1.npy"
training_audio_path = "/home/dl-group/data/Audio/audio1.npy"

print("Loading training data.")
training_vid = np.load(training_vid_path)
training_aud = np.load(training_audio_path)
# take the first video and make a matrix used as part of generator context
first_vid = np.zeros(training_vid.shape, dtype='float16')
for i in range(first_vid.shape[0]):
    first_vid[i,:,:,:] = training_vid[0,:,:,:]

training_vid.shape
training_aud.shape

#
# Training block
#


cnt = 1
num_epoch = 200
batch_size = 10
seed_size = 42
work_path = './'
save_freq = 10
metrics = []

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
    # Train discriminator on real and fake
    discriminator_metric_real = discriminator.train_on_batch([x_real_vid, x_real_aud], y_real)
    discriminator_metric_generated = discriminator.train_on_batch([x_fake_vid, x_real_aud], y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)
    # Train generator on Calculate losses
    # y_pred = discriminator.predict([x_real_aud, x_fake_vid])
    generator_metric = combination.train_on_batch([image_context, x_real_aud, x_real_aud], y_real)
    metrics.append([discriminator_metric, generator_metric])
    # Time for an update?
    # if epoch % save_freq == 0:
    #     save_images(cnt, fixed_seed)
    #     cnt += 1
    #  if epoch % save_freq == 0:   print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
    if epoch > 1 and epoch % save_freq == 0:
        generator.save("generator-e{}.h5".format(epoch))
        discriminator.save("discriminator-e{}.h5".format(epoch))

pd.DataFrame(np.asmatrix([[m[0][0] for m in metrics], [m[0][1] for m in metrics], [m[1] for m in metrics]]).T)




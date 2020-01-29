from tensorflow.keras.models import  load_model, Model
# load optimizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from model_build import build_discriminator, build_frame_discriminator, build_generator
# this checks the tf we are using
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

context_discriminator, aud_input = build_discriminator()
frame_discriminator, still_input = build_frame_discriminator()
#discriminator = load_model('./discriminator-e30.h5')
#discriminator.summary()
generator, inputs = build_generator()
#generator = load_model('./generator-e30.h5')
gd_joint = generator(inputs)

# The discriminator takes generated images as input and determines validity
context_validity = context_discriminator([gd_joint, aud_input])
frame_validity = frame_discriminator([gd_joint, still_input])

# For the combined model we will only train the generator
context_discriminator.trainable = False
frame_discriminator.trainable = False

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
context_inputs = inputs
frame_inputs = inputs
context_inputs.append(aud_input)
frame_inputs.append(still_input)
context_combination = Model(context_inputs, context_validity)
frame_combination = Model(frame_inputs, frame_validity)
context_combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
frame_combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


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
num_epoch = 1000
batch_size = 10
seed_size = 42
work_path = './'
save_freq = 100
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
    # Train context_discriminator on real and fake
    context_discriminator_metric_real = context_discriminator.train_on_batch([x_real_vid, x_real_aud], y_real)
    context_discriminator_metric_generated = context_discriminator.train_on_batch([x_fake_vid, x_real_aud], y_fake)
    context_discriminator_metric = 0.5 * np.add(context_discriminator_metric_real, context_discriminator_metric_generated)
    # Train generator on Calculate losses
    # y_pred = discriminator.predict([x_real_aud, x_fake_vid])
    generator_metric = context_combination.train_on_batch([image_context, x_real_aud, x_real_aud], y_real)
    metrics.append([context_discriminator_metric, generator_metric])
    # Time for an update?
    # if epoch % save_freq == 0:
    #     save_images(cnt, fixed_seed)
    #     cnt += 1
    #  if epoch % save_freq == 0:   print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
    print("I have epoched")
    if epoch > 1 and epoch % save_freq == 0:
        generator.save("generator-e{}.h5".format(epoch))
        context_discriminator.save("context_discriminator-e{}.h5".format(epoch))

generator.save("generator-e{}.h5".format(epoch))
context_discriminator.save("discriminator-e{}.h5".format(epoch))
pd.DataFrame(np.asmatrix([[m[0][0] for m in metrics], [m[0][1] for m in metrics], [m[1] for m in metrics]]).T)


import glob
import numpy as np
import random
from PIL import Image
from infogan import InfoGAN
from trainer import Trainer

def data_generator(batch_size, img_dir, img_size):
    w = img_size
    h = img_size
    image_filenames = glob.glob(img_dir + "/**/*.jpg")
    #import pdb;pdb.set_trace()
    counter = 0 
    while True:
        img_data = np.zeros((batch_size, w, h, 3))
        random.shuffle(image_filenames)
        if ((counter+1)*batch_size>=len(image_filenames)):
            counter = 0
        for i in range(batch_size):
            img = Image.open(image_filenames[counter + i]).resize((w, h))
            try:
                img_data[i] = np.array(img)
            except:
                print('failed on image:', image_filenames[counter+i])
                print(np.array(img).shape)
        yield img_data

image_size=256
batch_size = 64
data_generator = data_generator(batch_size, 'downloaded', image_size)
model = InfoGAN(input_shape=(image_size, image_size, 3), batch_size=batch_size)
model.gan.summary()
model.discriminator.summary()

trainer = Trainer(model)
trainer.fit_data_generator(data_generator, print_every=100)

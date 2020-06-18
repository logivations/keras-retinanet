import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('./snapshots', 'resnet50_pascal_10.h5')

# if the model is not converted to an inference model, use the line below
model = models.load_model(model_path, backbone_name='resnet50', convert=True)

# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet50')

folder_path = ''#os.getcwd()
labels_to_names = {0: 'head'}
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
relative_path_to_imgs = '/data/head-detection/HollywoodHeads/TestSet/'
relative_path_to_newimg = '/data/head-detection/HollywoodHeads/TestResults'
#os.mkdir('visual_results')
for i in os.listdir(relative_path_to_imgs):
    save_name = relative_path_to_newimg #i.split('/')[-1]
    image = read_image_bgr(relative_path_to_imgs + i)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        print(score)
        if score < 0.05:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    cv2.imwrite(save_name + i, draw[:, :, ::-1])
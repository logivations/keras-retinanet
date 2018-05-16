import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())
#
# print('tf.version', tf.__version__)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# exit()
model_path = os.path.join('./snapshots', '1', 'resnet50_pascal_31.h5')
# model_path = 'my_model.h5'
print (model_path)
# if the model is not converted to an inference model, use the line below
# see: https://github.com/logivations/keras-retinanet#converting-a-training-model-to-inference-model
model = models.load_model(model_path, backbone_name='resnet50', convert=True)

# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet50')

labels_to_names = {0: 'box'}
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
path = '/media/popovych/ae5cf79d-c161-4d60-8189-b57c694e3803/home/popovych/Documents/Projects/'
for i in os.listdir(path + '/keras-retinanet-master/test_3/JPEGImages/'):
    save_name = os.getcwd() + '/examples/1/result_' + i.split('/')[-1]
    image = read_image_bgr(path + '/keras-retinanet-master/test_3/JPEGImages/' + i)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    # print ('scale = ', scale)
    # print('scale = ', image.shape)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.98:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    cv2.imwrite(save_name, draw[:, :, ::-1])

    # create xml annotations
    # create_pascalvoc_annotation(i, all_boxes, output_folder="test_3/results/res_close_boxes/res_15/Annotations/")
    cv2.imwrite(save_name, draw[:, :, ::-1])
    # print(save_name)
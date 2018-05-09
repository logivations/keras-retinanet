import os
import keras
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())


model_path = os.path.join('snapshots/may7/', 'resnet50_pascal_09.h5')

model = keras.models.load_model(model_path, custom_objects=custom_objects)
labels_to_names = {0: 'none',
                   1: 'box'}

from keras_retinanet.bin.helps import create_pascalvoc_annotation

# for x in os.listdir(os.getcwd() + '/boxes_new/JPEGImages/'):
    # os.rename(os.getcwd() + '/boxes_new/JPEGImages/' + x,
    #           os.getcwd() + '/boxes_new/JPEGImages/' + x.replace(':', ''))
    # os.rename(os.getcwd() + '/boxes_new/JPEGImages/' + x,
    #           os.getcwd() + '/boxes_new/JPEGImages/' + x[:-4].replace('.', '') + '.jpg')
# exit()
for i in os.listdir(os.getcwd() + '/test_3/JPEGImages/'):
    save_name = os.getcwd() + '/test_3/results/res_train_all/result_' + i.split('/')[-1]
    image = read_image_bgr(os.getcwd() + '/test_3/JPEGImages/' + i)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    # print ('scale = ', scale)
    # print('scale = ', image.shape)

    start = time.time()
    _, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    predicted_labels = np.argmax(nms_classification[0, :, :], axis=1)
    scores = nms_classification[0, np.arange(nms_classification.shape[1]), predicted_labels]
    boxes /= scale

    all_boxes = []
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.5:
            continue
        color = label_color(label)
        b = boxes[0, idx, :].astype(int)
        # print('boxes', b)
        draw_box(draw, b, color=color)
        all_boxes.append(b)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    # print(all_boxes)
    # create xml annotations
    # create_pascalvoc_annotation(i, all_boxes, output_folder="test_3/results/res_close_boxes/res_15/Annotations/")
    cv2.imwrite(save_name, draw[:, :, ::-1])
    # print(save_name)
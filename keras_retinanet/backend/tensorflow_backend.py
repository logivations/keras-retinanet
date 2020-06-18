"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import keras


def map_fn(*args, **kwargs):
    return tf.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    return tf.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    return tf.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    return tf.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest' : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tf.image.ResizeMethod.BICUBIC,
        'area'    : tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize(images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    return tf.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tf.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return tf.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tf.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tf.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tf.compat.v1.where(*args, **kwargs)

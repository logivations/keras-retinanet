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

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr

import os
import numpy as np
from six import raise_from
from PIL import Image

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

voc_classes = {
    'head'   : 0
}


def _findNode(parent, name, debug_name = None, parse = None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class PascalVocGenerator(Generator):
    def __init__(
        self,
        data_dir,
        set_name,
        classes=voc_classes,
        image_extension='.jpeg',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path, self.set_name)

    def __parse_annotation(self, element):
        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return box

    def __parse_annotations(self, xml_root, img):

        try:
            size_node = _findNode(xml_root, 'size')
            width = _findNode(size_node, 'width', 'size.width', parse=float)
            height = _findNode(size_node, 'height', 'size.height', parse=float)
            filename = _findNode(xml_root, 'filename')
            #print(filename.text)
        except ValueError:
            with Image.open(img) as img:
                width, height = img.size

        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                box = self.__parse_annotation(element)
            except ValueError as e:
                print('could not parse object #{}: {}'.format(i, e))
                #raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)
            else:
                boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, image_index):
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot(), os.path.join(self.data_dir, 'JPEGImages', str(filename[:-3] + 'jpg')))
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
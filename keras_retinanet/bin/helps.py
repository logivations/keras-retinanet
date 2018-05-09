import xml.etree.cElementTree as ET


def create_pascalvoc_annotation(jpg_name, boxes, output_folder="path/to/Annotations/"):
    '''
    Generate xml annotation for dataset similar to pascalVOC.
    :param jpg_name: name of image
    :param boxes: list of bounding boxes for image
    :param output_folder: where to write annotation
    '''
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = jpg_name.split('/')[-1]
    for i in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "box"
        ET.SubElement(obj, "type").text = "bounding_box"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(i[0])
        ET.SubElement(bbox, "ymin").text = str(i[1])
        ET.SubElement(bbox, "xmax").text = str(i[2])
        ET.SubElement(bbox, "ymax").text = str(i[3])

        poly = ET.SubElement(obj, "polygon")
        for j in [[str(i[0]), str(i[1])], [str(i[2]), str(i[1])], [str(i[2]), str(i[3])], [str(i[0]), str(i[3])]]:
            pt = ET.SubElement(obj, "pt")
            ET.SubElement(pt, "x").text = j[0]
            ET.SubElement(pt, "y").text = j[1]
    tree = ET.ElementTree(root)

    tree.write(output_folder + jpg_name.split('/')[-1][:-4] + ".xml")

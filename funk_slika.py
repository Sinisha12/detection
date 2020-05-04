#!/usr/bin/env python
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util


def slika(image):
    def load_image_into_numpy_array(image1):
        (im_width, im_height) = image1.size
        return np.array(image1.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    sys.path.append("..")
    PATH_TO_CKPT = 'inference_graph/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('training', 'labelmap')
    NUM_CLASSES = 2
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            slika = Image.open(image)
            image_np = load_image_into_numpy_array(slika)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,  # True
                line_thickness=4,
                min_score_thresh=0.8,
                agnostic_mode=False)
            image_return = Image.fromarray(image_np)
            image_return.save("mysite/image_return2.jpg")
    return image_return

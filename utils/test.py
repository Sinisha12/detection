from utils import label_map_util
import os
NUM_CLASSES =2
PATH_TO_LABELS = os.path.join('training', 'labelmap.txt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(label_map)
print(categories)
print(category_index)
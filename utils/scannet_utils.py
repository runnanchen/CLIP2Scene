import os, sys
import csv
import json
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    import imageio
except:
    print("Please install the module 'imageio' for image processing, e.g.")
    print("pip install imageio")
    sys.exit(-1)
#g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']

# nyu40 label (1~40), 0 for unannotated, 41 for unknown
# only evaluate 20 classes in nyu40
CLASS_LABELS_scannet = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'other furniture']
CLASS_LABELS_S3DIS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookshelf', 'board', 'clutter']

VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
#UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1   # scannet github
UNKNOWN_ID = 41
UNANNOTATE_ID = 0

# only evaluate 20 classes in nyu40
# map nyu40 to 1~21, 0 for unannotated and unknown 
g_label_names = ['unannotate'] + CLASS_LABELS_scannet
g_label_ids = [UNANNOTATE_ID] + VALID_CLASS_IDS

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


# input: scene_types.txt or scene_types_all.txt
def read_scene_types_mapping(filename, remove_spaces=True):
    assert os.path.isfile(filename)
    mapping = dict()
    lines = open(filename).read().splitlines()
    lines = [line.split('\t') for line in lines]
    if remove_spaces:
        mapping = { x[1].strip():int(x[0]) for x in lines }
    else:
        mapping = { x[1]:int(x[0]) for x in lines }        
    return mapping


# color by label
def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)


# color by different instances (mod length of color palette)
def visualize_instance_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    instances = np.unique(image)
    for idx, inst in enumerate(instances):
        vis_image[image==inst] = color_palette[inst%len(color_palette)]
    imageio.imwrite(filename, vis_image)


def create_color_palette():
    return [
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (247, 182, 210),		# desk
       (219, 219, 141),		# curtain
       (255, 127, 14), 		# refrigerator
       (158, 218, 229),		# shower curtain
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (227, 119, 194),		# bathtub
       (82, 84, 163),  		# otherfurn
        (0, 0, 0),
    ]

# color palette for nyu40 labels
# def create_color_palette():
#     return [
#        (0, 0, 0),
#        (174, 199, 232),		# wall
#        (152, 223, 138),		# floor
#        (31, 119, 180), 		# cabinet
#        (255, 187, 120),		# bed
#        (188, 189, 34), 		# chair
#        (140, 86, 75),  		# sofa
#        (255, 152, 150),		# table
#        (214, 39, 40),  		# door
#        (197, 176, 213),		# window
#        (148, 103, 189),		# bookshelf
#        (196, 156, 148),		# picture
#        (23, 190, 207), 		# counter
#        (178, 76, 76),
#        (247, 182, 210),		# desk
#        (66, 188, 102),
#        (219, 219, 141),		# curtain
#        (140, 57, 197),
#        (202, 185, 52),
#        (51, 176, 203),
#        (200, 54, 131),
#        (92, 193, 61),
#        (78, 71, 183),
#        (172, 114, 82),
#        (255, 127, 14), 		# refrigerator
#        (91, 163, 138),
#        (153, 98, 156),
#        (140, 153, 101),
#        (158, 218, 229),		# shower curtain
#        (100, 125, 154),
#        (178, 127, 135),
#        (120, 185, 128),
#        (146, 111, 194),
#        (44, 160, 44),  		# toilet
#        (112, 128, 144),		# sink
#        (96, 207, 209),
#        (227, 119, 194),		# bathtub
#        (213, 92, 176),
#        (94, 106, 211),
#        (82, 84, 163),  		# otherfurn
#        (100, 85, 144)
#     ]

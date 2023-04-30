import os
import sys
import time
import argparse
import json
import numpy as np
import multiprocessing as mp
from functools import partial

# sys.path.append("..")
sys.path.append("../utils")
import pc_utils
import scannet_utils
from plyfile import PlyData, PlyElement

g_label_names = scannet_utils.g_label_names
g_label_ids = scannet_utils.g_label_ids

''' 
    params 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', default= os.environ['HOME']+'/dataset/scannet/scans/scans',
                    help='path to scannet data')
parser.add_argument('--label_map_file', default= os.environ['HOME'] + '/dataset/scannet/scans/scannetv2-labels.combined.tsv',
                    help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument("--num_proc", required=False, type=int, default=28, help="number of parallel process, default is 30")
opt = parser.parse_args()

remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i


def collect_point_data(scene_name):
    # read label mapping file
    label_map = scannet_utils.read_label_mapping(opt.label_map_file, label_from='raw_category', label_to='nyu40id')
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(opt.scannet_path, scene_name)
    out_filename = os.path.join(data_folder, scene_name + '_new_semantic.npy')  # scene0000_00/scene0000_00.npy
    # if os.path.exists(out_filename): return
    # Read segmentation label
    seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))
    seg_to_verts, num_verts = scannet_utils.read_segmentation(seg_filename)
    # Read Instances segmentation label
    # agg_filename = os.path.join(data_folder, '%s.aggregation.json' % (scene_name))
    # object_id_to_segs, label_to_segs = scannet_utils.read_aggregation(agg_filename)
    # Raw points in XYZRGBA
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    label_filename = os.path.join(data_folder, '%s_vh_clean_2.labels.ply' % (scene_name))
    points = pc_utils.read_ply_rgba_normal(ply_filename)
    # points = pc_utils.read_ply_rgba(ply_filename)
    # labels = pc_utils.read_ply_rgba(label_filename)

    # plydata = PlyData.read(label_filename)
    # pc = plydata['vertex'].data
    # pc_array = np.array([[l] for x,y,z,r,g,b,a, l in pc])
    # trans_ids = np.array([[g_label_ids.index(l)] for x,y,z,r,g,b,a, l in pc])

    plydata = PlyData().read(label_filename)
    labels = np.expand_dims(remapper[np.array(plydata.elements[0]['label'])],1)


    # trans_ids = g_label_ids.index(pc_array)

    # import pdb
    # pdb.set_trace()
    '''
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for label, segs in label_to_segs.items():
        # convert scannet raw label to nyu40 label (1~40), 0 for unannotated, 41 for unknown
        label_id = label_map[label]

        # only evaluate 20 class in nyu40 label
        # map nyu40 to 1~21, 0 for unannotated, unknown and not evalutated
        if label_id in g_label_ids:  # IDS for 20 classes in nyu40 for evaluation (1~21)
            eval_label_id = g_label_ids.index(label_id)
        else:  # IDS unannotated, unknow or not for evaluation go to unannotate label (0)
            eval_label_id = g_label_names.index('unannotate')
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = eval_label_id
    '''
    # for i in range(20):
    #    print(label_ids[i])

    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id

    for i in range(max(instance_ids)):
        index = instance_ids == i
        min_label = min(labels[index])
        max_label = max(labels[index])
        if min_label != max_label: print("error")

    points = np.delete(points, 6, 1)  # only RGB, ignoring A
    # label_ids = np.expand_dims(label_ids, 1)
    # instance_ids = np.expand_dims(instance_ids, 1)
    # print(points.shape, label_ids.shape, instance_ids.shape)
    # order is critical, do not change the order
    # print("labels data: ", label_ids - labels)

    # data = np.concatenate((points, labels, labels), 1)
    data = np.concatenate((points, instance_ids, labels), 1)
    # data = np.concatenate((points, instance_ids, label_ids), 1)
    print(out_filename)
    if os.path.exists(out_filename): return
    np.save(out_filename, data)
    # print(scene_name, ' points shape:', data.shape)


def preprocess_scenes(scene_name):
    try:
        collect_point_data(scene_name)
        print("name: ", scene_name)
    except Exception as e:
        sys.stderr.write(scene_name + 'ERROR!!')
        sys.stderr.write(str(e))
        sys.exit(-1)

def main():
    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    scenes.sort()
    # collect_point_data(scenes[10])
    # import pdb
    # pdb.set_trace()
    print(opt.scannet_path)
    print('Find %d scenes' % len(scenes))
    print('Extract points (Vertex XYZ, RGB, NxNyNz, Label, Instance-label)')

    pool = mp.Pool(opt.num_proc)
    pool.map(preprocess_scenes, scenes)


if __name__ == '__main__':
    main()

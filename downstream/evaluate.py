import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
# from torchsparse import SparseTensor
from utils.metrics import compute_IoU


CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

CLASSES_scannet = [
    'wall',
    'floor',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'desk',
    'curtain',
    'refrigerator',
    'shower curtain',
    'toilet',
    'sink',
    'bathtub',
    'other furniture'
]

def evaluate(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            lidar_names = batch["lidar_name"]

            sparse_input = SparseTensor(batch["sinput_F"].float(), batch["sinput_C"].int(), device=0)
            # print(sparse_input, model)
            output_points = model(sparse_input)

            # for spvcnn
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
            # output_points = model(sparse_input.to(0))
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0

            # print(output_points)
            # print(batch["evaluation_labels"][0].max())
            # print(batch["evaluation_labels"][0].min())


            for j, lb in enumerate(batch["len_batch"]):
                # print(batch["len_batch"], j)
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # print(predictions.shape, batch["evaluation_labels"][j].shape)
                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb

                # m_IoU, fw_IoU, per_class_IoU = compute_IoU(
                #     torch.cat([predictions]),
                #     torch.cat([deepcopy(batch["evaluation_labels"][j])]),
                #     config["model_n_out"],
                #     ignore_index=0,
                # )

                '''
                class_ind = 4
                lidar_name = lidar_names[j].split('/')[-1]
                root_path = '/mnt/lustre/chenrunnan/projects/SLidR/visual/annotation_free/'
                # lidar_name_path = root_path + str(per_class_IoU[class_ind]) + lidar_name
                lidar_name_path = root_path + lidar_name
                save_file = predictions.unsqueeze(-1).numpy()
                # save_file = np.expand_dims(predictions)
                # if per_class_IoU[class_ind] != 1 and per_class_IoU[class_ind] > 0.4:
                np.array(save_file).astype(np.uint8).tofile(lidar_name_path)
                '''

                # import pdb
                # pdb.set_trace()

            i += j


        full_predictions = torch.cat(full_predictions).int()
        ground_truth = torch.cat(ground_truth).int()

        # if config["dataset"].lower() == "scannet":
        #     ground_truth += 1
        #     ground_truth[ground_truth == -99] = 0

        # print(full_predictions.shape, torch.cat(ground_truth).shape)
        # print(torch.cat(full_predictions), torch.cat(ground_truth))

        print(ground_truth)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            full_predictions,
            ground_truth,
            config["model_n_out"],
            ignore_index=0,
        )

        # import pdb
        # pdb.set_trace()

        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "scannet":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_scannet, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU

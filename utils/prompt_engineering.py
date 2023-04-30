import numpy as np
import torch
import clip
import argparse

scannet_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'other furniture']

nuscenes_classes = ["barrier", "bicycle", "bus", "car", "construction vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "driveable surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]

kitti_classes = [ "car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic sign"]

cityscapes_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

ade20k_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

coco_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor', 'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural', 'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window', 'window', 'wood']

voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

pascal_context_classes = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']

all_pascal_context_classes = ['accordion', 'airplane', 'air conditioner', 'antenna', 'artillery', 'ashtray', 'atrium', 'baby carriage', 'bag', 'ball', 'balloon', 'bamboo weaving', 'barrel', 'baseball bat', 'basket', 'basketball backboard', 'bathtub', 'bed', 'bedclothes', 'beer', 'bell', 'bench', 'bicycle', 'binoculars', 'bird', 'bird cage', 'bird feeder', 'bird nest', 'blackboard', 'board', 'boat', 'bone', 'book', 'bottle', 'bottle opener', 'bowl', 'box', 'bracelet', 'brick', 'bridge', 'broom', 'brush', 'bucket', 'building', 'bus', 'cabinet', 'cabinet door', 'cage', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camera lens', 'can', 'candle', 'candle holder', 'cap', 'car', 'card', 'cart', 'case', 'casette recorder', 'cash register', 'cat', 'cd', 'cd player', 'ceiling', 'cell phone', 'cello', 'chain', 'chair', 'chessboard', 'chicken', 'chopstick', 'clip', 'clippers', 'clock', 'closet', 'cloth', 'clothes tree', 'coffee', 'coffee machine', 'comb', 'computer', 'concrete', 'cone', 'container', 'control booth', 'controller', 'cooker', 'copying machine', 'coral', 'cork', 'corkscrew', 'counter', 'court', 'cow', 'crabstick', 'crane', 'crate', 'cross', 'crutch', 'cup', 'curtain', 'cushion', 'cutting board', 'dais', 'disc', 'disc case', 'dishwasher', 'dock', 'dog', 'dolphin', 'door', 'drainer', 'dray', 'drink dispenser', 'drinking machine', 'drop', 'drug', 'drum', 'drum kit', 'duck', 'dumbbell', 'earphone', 'earrings', 'egg', 'electric fan', 'electric iron', 'electric pot', 'electric saw', 'electronic keyboard', 'engine', 'envelope', 'equipment', 'escalator', 'exhibition booth', 'extinguisher', 'eyeglass', 'fan', 'faucet', 'fax machine', 'fence', 'ferris wheel', 'fire extinguisher', 'fire hydrant', 'fire place', 'fish', 'fish tank', 'fishbowl', 'fishing net', 'fishing pole', 'flag', 'flagstaff', 'flame', 'flashlight', 'floor', 'flower', 'fly', 'foam', 'food', 'footbridge', 'forceps', 'fork', 'forklift', 'fountain', 'fox', 'frame', 'fridge', 'frog', 'fruit', 'funnel', 'furnace', 'game controller', 'game machine', 'gas cylinder', 'gas hood', 'gas stove', 'gift box', 'glass', 'glass marble', 'globe', 'glove', 'goal', 'grandstand', 'grass', 'gravestone', 'ground', 'guardrail', 'guitar', 'gun', 'hammer', 'hand cart', 'handle', 'handrail', 'hanger', 'hard disk drive', 'hat', 'hay', 'headphone', 'heater', 'helicopter', 'helmet', 'holder', 'hook', 'horse', 'horse-drawn carriage', 'hot-air balloon', 'hydrovalve', 'ice', 'inflator pump', 'ipod', 'iron', 'ironing board', 'jar', 'kart', 'kettle', 'key', 'keyboard', 'kitchen range', 'kite', 'knife', 'knife block', 'ladder', 'ladder truck', 'ladle', 'laptop', 'leaves', 'lid', 'life buoy', 'light', 'light bulb', 'lighter', 'line', 'lion', 'lobster', 'lock', 'machine', 'mailbox', 'mannequin', 'map', 'mask', 'mat', 'match book', 'mattress', 'menu', 'metal', 'meter box', 'microphone', 'microwave', 'mirror', 'missile', 'model', 'money', 'monkey', 'mop', 'motorbike', 'mountain', 'mouse', 'mouse pad', 'musical instrument', 'napkin', 'net', 'newspaper', 'oar', 'ornament', 'outlet', 'oven', 'oxygen bottle', 'pack', 'pan', 'paper', 'paper box', 'paper cutter', 'parachute', 'parasol', 'parterre', 'patio', 'pelage', 'pen', 'pen container', 'pencil', 'person', 'photo', 'piano', 'picture', 'pig', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plastic', 'plate', 'platform', 'player', 'playground', 'pliers', 'plume', 'poker', 'poker chip', 'pole', 'pool table', 'postcard', 'poster', 'pot', 'potted plant', 'printer', 'projector', 'pumpkin', 'rabbit', 'racket', 'radiator', 'radio', 'rail', 'rake', 'ramp', 'range hood', 'receiver', 'recorder', 'recreational machines', 'remote control', 'road', 'robot', 'rock', 'rocket', 'rocking horse', 'rope', 'rug', 'ruler', 'runway', 'saddle', 'sand', 'saw', 'scale', 'scanner', 'scissors', 'scoop', 'screen', 'screwdriver', 'sculpture', 'scythe', 'sewer', 'sewing machine', 'shed', 'sheep', 'shell', 'shelves', 'shoe', 'shopping cart', 'shovel', 'sidecar', 'sidewalk', 'sign', 'signal light', 'sink', 'skateboard', 'ski', 'sky', 'sled', 'slippers', 'smoke', 'snail', 'snake', 'snow', 'snowmobiles', 'sofa', 'spanner', 'spatula', 'speaker', 'speed bump', 'spice container', 'spoon', 'sprayer', 'squirrel', 'stage', 'stair', 'stapler', 'stick', 'sticky note', 'stone', 'stool', 'stove', 'straw', 'stretcher', 'sun', 'sunglass', 'sunshade', 'surveillance camera', 'swan', 'sweeper', 'swim ring', 'swimming pool', 'swing', 'switch', 'table', 'tableware', 'tank', 'tap', 'tape', 'tarp', 'telephone', 'telephone booth', 'tent', 'tire', 'toaster', 'toilet', 'tong', 'tool', 'toothbrush', 'towel', 'toy', 'toy car', 'track', 'train', 'trampoline', 'trash bin', 'tray', 'tree', 'tricycle', 'tripod', 'trophy', 'truck', 'tube', 'turtle', 'tv monitor', 'tweezers', 'typewriter', 'umbrella', 'unknown', 'vacuum cleaner', 'vending machine', 'video camera', 'video game console', 'video player', 'video tape', 'violin', 'wakeboard', 'wall', 'wallet', 'wardrobe', 'washing machine', 'watch', 'water', 'water dispenser', 'water pipe', 'water skate board', 'watermelon', 'whale', 'wharf', 'wheel', 'wheelchair', 'window', 'window blinds', 'wineglass', 'wire', 'wood', 'wool']

bg_classes = ['building', 'ground', 'grass', 'tree', 'sky']

mickey_classes = ['Mickey Mouse', 'Donald Duck'] + bg_classes
batman_classes = ['Batman', 'Joker'] + bg_classes
mario_classes = ['Mario', 'Luigi'] + bg_classes
gates_classes = ['Bill Gates', 'Steve Jobs'] + bg_classes

cityscapes_no_person_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
batman_ext_classes = ['Batman', 'Joker', 'James Gordon', 'The Penguin', 'Robin', 'Alfred Pennyworth', 'Catwoman', 'Harley Quinn'] + cityscapes_no_person_classes

sports_classes = ['baseball player', 'basketball player', 'soccer player', 'football player', 'person', 'background', 'wall', 'building', 'sky', 'grass', 'tree', 'ground', 'floor', 'baseball court', 'basketball court', 'soccer court', 'football court']
car_brands_classes = ['Bugatti', 'Cadillac', 'Porsche', 'Lamborghini', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
blur_classes = ['very blurry car', 'car', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
car_color_classes = ['white car', 'blue car', 'red car', 'black car', 'green car', 'yellow car', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

prompt_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',
]

def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engeering script')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='clip model name')
    parser.add_argument('--class-set', default=['voc'], nargs='+',
        choices=['kitti', 'nuscenes', 'scannet', 'city', 'ade', 'stuff', 'voc', 'context', 'acontext', 'mickey', 'batman', 'mario', 'gates', 'blur', 'sports', 'car_brands', 'batman_ext', 'car_color'],
        help='the set of class names')
    parser.add_argument('--no-prompt-eng', action='store_true', help='disable prompt engineering')

    args = parser.parse_args()
    return args

def zeroshot_classifier(model_name, classnames, templates):
    model, preprocess = clip.load(model_name)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

if __name__ == '__main__':
    args = parse_args()

    classes = []
    all_set_name = ''
    name_mapping = {'kitti': kitti_classes, 'nuscenes': nuscenes_classes, 'scannet': scannet_classes, 'city': cityscapes_classes, 'ade': ade20k_classes, 'stuff': coco_stuff_classes, 'voc': voc_classes, 'context': pascal_context_classes, 'acontext': all_pascal_context_classes, 'mickey': mickey_classes, 'batman': batman_classes, 'mario': mario_classes, 'gates': gates_classes, 'blur': blur_classes, 'sports': sports_classes, 'car_brands': car_brands_classes, 'batman_ext': batman_ext_classes, 'car_color': car_color_classes}
    for set_name in args.class_set:
        if set_name in name_mapping:
            classes += name_mapping[set_name]
            all_set_name += '_{}'.format(set_name)
        if set_name in ['blur'] or args.no_prompt_eng:
            prompt_templates = ['a photo of a {}.']
    # remove redundant classes
    classes = list(dict.fromkeys(classes))
    # remove the first underline
    all_set_name = all_set_name[1:]
    print(classes)

    print(f"{len(classes)} class(es), {len(prompt_templates)} template(s)")

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16'}
    zeroshot_weights = zeroshot_classifier(name_mapping[args.model], classes, prompt_templates)
    zeroshot_weights = zeroshot_weights.permute(1, 0).float()
    print(zeroshot_weights.shape)

    prefix = f'{all_set_name}_{args.model}'
    if args.no_prompt_eng:
        prefix += '_npe'

    torch.save(zeroshot_weights, f'{prefix}_clip_text.pth')


import torch
import clip
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save the CLIP visual weights')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT32', 'ViT16', 'ViT14'], help='clip model name')
    parser.add_argument('--backbone', action='store_true', help='Prepend the word backbone to the key so that it can be directly loaded as a checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', \
        'RN50x16': 'RN50x16', 'RN50x64': 'RN50x64', \
        'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16', 'ViT14': 'ViT-L/14'}
    clip_model, preprocess = clip.load(name_mapping[args.model], device='cpu')
    state_dict = clip_model.state_dict()

    result_model = {'meta': {}, 'state_dict': {}}
    all_model = dict()
    stem_mapping = {'conv1': 0, 'bn1': 1, 'conv2': 3, 'bn2': 4, 'conv3': 6, 'bn3':7}
    clip_keys = []
    prefix = 'visual'
    for key in state_dict.keys():
        if 'ViT' in args.model and prefix in key:
            new_key = key[len(f'{prefix}.'):]
            if new_key == 'proj':
                all_model['proj'] = {}
                all_model['proj']['weight'] = state_dict[key].float().t()
                continue
            if new_key == 'class_embedding':
                new_key = 'cls_token'
                state_dict[key] = state_dict[key][None, None, :]
            elif new_key == 'positional_embedding':
                new_key = 'pos_embed'
                state_dict[key] = state_dict[key][None, :, :]
            elif new_key == 'conv1.weight':
                new_key = 'patch_embed.projection.weight'
            elif 'ln_pre' in new_key:
                weight_or_bias = new_key.split('.')[-1]
                new_key = f'ln0.{weight_or_bias}'
            elif 'ln_post' in new_key:
                weight_or_bias = new_key.split('.')[-1]
                new_key = f'ln1.{weight_or_bias}'
            elif 'transformer' in new_key:
                new_key = 'layers.' + new_key[len('transformer.resblocks.'):]
                if 'mlp' in new_key:
                    new_key = new_key.replace('mlp', 'ffn.layers')
                if 'c_fc' in new_key:
                    new_key = new_key.replace('c_fc', '0.0')
                if 'c_proj' in new_key:
                    new_key = new_key.replace('c_proj', '1')
                if 'attn' in new_key:
                    new_key = new_key.replace('attn', 'attn.attn')
                elif 'ln_' in new_key:
                    new_key = new_key.replace('ln_', 'ln')
            if args.backbone:
                new_key = 'backbone.' + new_key
            clip_keys.append(new_key)
            result_model['state_dict'].update({new_key: state_dict[key].float()})
        elif prefix in key:
            if 'attnpool' in key:
                if 'proj' in key:
                    proj_name = key.split('.')[2]
                    weight_or_bias = key.split('.')[3]
                    if proj_name not in all_model:
                        all_model[proj_name] = {}
                    all_model[proj_name][weight_or_bias] = state_dict[key].float()
            else:
                new_key = key[len(f'{prefix}.'):]
                if 'layer' not in new_key:
                    layer_name, layer_type = new_key.split('.')
                    new_key = 'stem.{}.{}'.format(stem_mapping[layer_name], layer_type)
                if 'downsample' in new_key:
                    splits = new_key.split('.')
                    new_key = '{}.{}.{}.{}.{}'.format(splits[0], splits[1], splits[2], \
                        int(splits[3])+1, splits[4])
                if args.backbone:
                    new_key = 'backbone.' + new_key
                clip_keys.append(new_key)
                result_model['state_dict'].update({new_key: state_dict[key].float()})

    if args.backbone:
        torch.save(result_model, f'{args.model}_clip_backbone.pth')
    else:
        all_model['clip'] = result_model['state_dict']
        torch.save(all_model, '{}_clip_weights.pth'.format(args.model))
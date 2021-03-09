# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse
import numpy as np
from collections import defaultdict
import json
import pickle
import os

import vsrl_utils as vu


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, required=True,
    )
    parser.add_argument(
        '--prior_path', type=str, required=True,
    )
    parser.add_argument(
        '--save_path', type=str, required=True,
    )

    args = parser.parse_args()

    return args


def set_hoi(box_annotations, hoi_annotations, verb_classes):
    no_object_id = -1

    hoia_annotations = defaultdict(lambda: {
        'annotations': [],
        'hoi_annotation': []
    })

    for action_annotation in hoi_annotations:
        for label, img_id, role_ids in zip(action_annotation['label'][:, 0],
                                           action_annotation['image_id'][:, 0],
                                           action_annotation['role_object_id']):
            hoia_annotations[img_id]['file_name'] = box_annotations[img_id]['file_name']
            hoia_annotations[img_id]['annotations'] = box_annotations[img_id]['annotations']

            if label == 0:
                continue

            subject_id = box_annotations[img_id]['annotation_ids'].index(role_ids[0])

            if len(role_ids) == 1:
                hoia_annotations[img_id]['hoi_annotation'].append(
                    {'subject_id': subject_id, 'object_id': no_object_id,
                     'category_id': verb_classes.index(action_annotation['action_name'])})
                continue

            for role_name, role_id in zip(action_annotation['role_name'][1:], role_ids[1:]):
                if role_id == 0:
                    object_id = no_object_id
                else:
                    object_id = box_annotations[img_id]['annotation_ids'].index(role_id)

                hoia_annotations[img_id]['hoi_annotation'].append(
                    {'subject_id': subject_id, 'object_id': object_id,
                     'category_id': verb_classes.index('{}_{}'.format(action_annotation['action_name'], role_name))})

    hoia_annotations = [v for v in hoia_annotations.values()]

    return hoia_annotations


def main(args):
    vsgnet_verbs_classes = {
        'carry_obj': 0,
        'catch_obj': 1,
        'cut_instr':2,
        'cut_obj': 3,
        'drink_instr': 4,
        'eat_instr':5,
        'eat_obj': 6,
        'hit_instr':7,
        'hit_obj': 8,
        'hold_obj': 9,
        'jump_instr': 10,
        'kick_obj': 11,
        'lay_instr': 12,
        'look_obj': 13,
        'point_instr': 14,
        'read_obj': 15,
        'ride_instr': 16,
        'run': 17,
        'sit_instr': 18,
        'skateboard_instr': 19,
        'ski_instr': 20,
        'smile': 21,
        'snowboard_instr': 22,
        'stand': 23,
        'surf_instr': 24,
        'talk_on_phone_instr': 25,
        'throw_obj': 26,
        'walk': 27,
        'work_on_computer_instr': 28
    }

    box_annotations = defaultdict(lambda: {
        'annotations': [],
        'annotation_ids': []
    })

    coco = vu.load_coco(args.load_path)

    img_ids = coco.getImgIds()
    img_infos = coco.loadImgs(img_ids)

    for img_info in img_infos:
        box_annotations[img_info['id']]['file_name'] = img_info['file_name']

    annotation_ids = coco.getAnnIds(imgIds=img_ids)
    annotations = coco.loadAnns(annotation_ids)
    for annotation in annotations:
        img_id = annotation['image_id']
        category_id = annotation['category_id']
        box = np.array(annotation['bbox'])
        box[2:] += box[:2]

        box_annotations[img_id]['annotations'].append({'category_id': category_id, 'bbox': box.tolist()})
        box_annotations[img_id]['annotation_ids'].append(annotation['id'])

    hoi_trainval = vu.load_vcoco('vcoco_trainval')
    hoi_test = vu.load_vcoco('vcoco_test')

    action_classes = [x['action_name'] for x in hoi_trainval]
    verb_classes = []
    for action in hoi_trainval:
        if len(action['role_name']) == 1:
            verb_classes.append(action['action_name'])
        else:
            verb_classes += ['{}_{}'.format(action['action_name'], r) for r in action['role_name'][1:]]

    print('Verb class')
    for i, verb_class in enumerate(verb_classes):
        print('{:02d}: {}'.format(i, verb_class))

    hoia_trainval_annotations = set_hoi(box_annotations, hoi_trainval, verb_classes)
    hoia_test_annotations = set_hoi(box_annotations, hoi_test, verb_classes)

    print('#Training images: {}, #Test images: {}'.format(len(hoia_trainval_annotations), len(hoia_test_annotations)))

    with open(os.path.join(args.save_path, 'trainval_vcoco.json'), 'w') as f:
        json.dump(hoia_trainval_annotations, f)

    with open(os.path.join(args.save_path, 'test_vcoco.json'), 'w') as f:
        json.dump(hoia_test_annotations, f)

    with open(args.prior_path, 'rb') as f:
        prior = pickle.load(f)

    prior = [prior[k] for k in sorted(prior.keys())]
    prior = np.concatenate(prior).T
    prior = prior[[vsgnet_verbs_classes[verb_class] for verb_class in verb_classes]]
    np.save(os.path.join(args.save_path, 'corre_vcoco.npy'), prior)


if __name__ == '__main__':
    args = get_args()
    main(args)

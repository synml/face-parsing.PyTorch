import argparse
import os

import cv2
import numpy as np
import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    face_data = os.path.join(args.dataset_dir, 'CelebAMask-HQ-img')
    face_sep_mask = os.path.join(args.dataset_dir, 'CelebAMask-HQ-mask-anno')
    output_mask_path = os.path.join(args.dataset_dir, 'mask')
    os.makedirs(output_mask_path, exist_ok=True)

    attributes = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                  'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    counter = 0
    total = 0
    for i in tqdm.tqdm(range(15), desc='Preprocess'):
        for j in tqdm.tqdm(range(i * 2000, (i + 1) * 2000), 'Subset', leave=False):
            mask = np.zeros((512, 512))
            for index, attr in enumerate(attributes, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', attr, '.png'])
                path = os.path.join(face_sep_mask, str(i), file_name)
                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))

                    mask[sep_mask == 225] = index
            cv2.imwrite(os.path.join(output_mask_path, f'{j}.png'), mask)
    print(f'{counter=}, {total=}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

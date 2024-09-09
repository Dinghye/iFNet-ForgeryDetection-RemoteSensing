# -*- coding:utf-8 -*-
'''
Preprocessed dataset.

Version 1.0  2024-04-08 20:35:44 by QiJi.
TODO:
1. xxx

'''

import os
import numpy as np
import shutil
# import random

RD_SEED = 2024
TRAIN_VAL_RATIO = [0.5, 0.5]


def split_dataset(src_dir, dst_dir, ratio=[0.5, 0.5]):
    img_dir = src_dir+'/image'
    lbl_file_path = src_dir+'/gt.txt'

    # 读取标签文件并解析内容
    img_list = []
    lbl_list = []
    with open(lbl_file_path, 'r') as file:
        for line in file:
            # 分割每行以获取影像名称和对应的标签
            img_name, lbl = line.strip().split()

            img_list.append(img_name)
            lbl_list.append(int(lbl))

    img_list = np.array(img_list)
    lbl_list = np.array(lbl_list)

    # Split 0_real and 1_fake
    real_img_list = img_list[lbl_list == 0]
    fake_img_list = img_list[lbl_list == 1]

    #  Split train and val
    real_train_num = round(len(real_img_list) * ratio[0])
    fake_train_num = round(len(fake_img_list) * ratio[0])

    np.random.seed(RD_SEED)
    np.random.shuffle(real_img_list)
    np.random.seed(RD_SEED)
    np.random.shuffle(fake_img_list)

    train_dir = dst_dir + '/train'
    val_dir = dst_dir + '/val'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Train - Real
    if not os.path.exists(train_dir+'/0_real'):
        os.makedirs(train_dir+'/0_real')
    for name in real_img_list[:real_train_num]:
        src_img_pth = img_dir+'/'+name
        dst_img_pth = train_dir+'/0_real/'+name
        shutil.copyfile(src_img_pth, dst_img_pth)
    print('Train - Real: ', len(real_img_list[:real_train_num]))

    if not os.path.exists(train_dir+'/1_fake'):
        os.makedirs(train_dir+'/1_fake')
    for name in fake_img_list[:fake_train_num]:
        src_img_pth = img_dir+'/'+name
        dst_img_pth = train_dir+'/1_fake/'+name
        shutil.copyfile(src_img_pth, dst_img_pth)
    print('Train - Fake: ', len(fake_img_list[:fake_train_num]))

    if not os.path.exists(val_dir+'/0_real'):
        os.makedirs(val_dir+'/0_real')
    for name in real_img_list[real_train_num:]:
        src_img_pth = img_dir+'/'+name
        dst_img_pth = val_dir+'/0_real/'+name
        shutil.copyfile(src_img_pth, dst_img_pth)
    print('Val - Real: ', len(real_img_list[real_train_num:]))

    if not os.path.exists(val_dir+'/1_fake'):
        os.makedirs(val_dir+'/1_fake')
    for name in fake_img_list[fake_train_num:]:
        src_img_pth = img_dir+'/'+name
        dst_img_pth = val_dir+'/1_fake/'+name
        shutil.copyfile(src_img_pth, dst_img_pth)
    print('Val - Fake: ', len(fake_img_list[fake_train_num:]))


def split_dataset2(src_dir, dst_dir, ratio=[0.5, 0.5]):

    for cls in ['0_real', '1_fake']:
        src_cls_dir = f'{src_dir}/{cls}'
        dst_cls_dir = f'{dst_dir}/{cls}'
        os.makedirs(dst_cls_dir, exist_ok=True)

        img_list = sorted(os.listdir(src_cls_dir))
        train_num = round(len(img_list) * ratio[0])
        np.random.seed(RD_SEED)
        np.random.shuffle(img_list)

        for name in img_list[train_num:]:
            src_img_pth = src_cls_dir+'/'+name
            dst_img_pth = dst_cls_dir+'/'+name
            shutil.move(src_img_pth, dst_img_pth)


if __name__ == "__main__":
    # root = r'N:\Classification\ISPRS_RSI_Authentication'
    # org_dir = root + '/Original/train_set'

    # split_dataset(org_dir, root)

    root = r'D:\Classification\SDGen-Detection\SD_Potsdam'
    train_dir = root + '/train'
    val_dir = root + '/val'
    split_dataset2(train_dir, val_dir)

    print('...')

import os
import itertools
import yaml
from copy import deepcopy
import argparse
import math
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
import nibabel as nib

import contours as C

# Default Settings
XY_CLASS = [0,3,4,5,6,7,8,9]
Z_CLASS = ["prism", "hourglass", "pyramid", "rhombus"]
ROTATE_L = ["stay", "side1", "side2", "side3", "side4", "upsidedown"]
NUM_PER_IMAGE = 20
NOISE_RANGE_DICT = {0:(0,0), 1:(125,200), 2:(300, 400)}
WORKER_NUM = 1

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, required=True)
parser.add_argument('--worker_id', type=int, default=[i for i in range(WORKER_NUM)])
parser.add_argument('--base_save_dir', type=str, required=True)
args = parser.parse_args()

# Update Args to Yaml Setting
with open(args.yaml_path) as file:
    obj = yaml.safe_load(file)
for k,v in obj.items():
    setattr(args, k, v)

# Additional setting
SPACE = (args.space, args.space, args.space)
args.noise_range = NOISE_RANGE_DICT[args.noise_strength]
if args.no_rotate:
    ROTATE_L = ["stay"]
if args.reduce_xy_class:
    XY_CLASS = [0,3,4,5,6]
if args.reduce_z_class:
    Z_CLASS = ["prism", "hourglass"]


def mk_savedir():
    save_name = args.yaml_path.split('/')[-1].split('.')[0]
    exp_save_dir = os.path.join(args.base_save_dir, save_name)
    os.makedirs(f"{exp_save_dir}/img", exist_ok=True)
    os.makedirs(f"{exp_save_dir}/label", exist_ok=True)
    if args.set_lost_label:
        os.makedirs(f"{exp_save_dir}/label_lostxy", exist_ok=True)
        os.makedirs(f"{exp_save_dir}/label_lostz", exist_ok=True)
    args.save_dir = exp_save_dir


def class2function(xy_rule, z_rule):
    if xy_rule == 0: # if the slice is ellipse
        if z_rule == "prism":
            return "Cylinder"
        elif z_rule == "hourglass":
            return "Hourglass_Circle"
        elif z_rule == "pyramid":
            return "Cone"
        elif z_rule == "rhombus":
            return "Rhombus_Circle"
        elif z_rule == "sphere":
            return "Sphere"
        else:
            raise ValueError()
    else:
        if z_rule == "prism":
            return "Prism"
        elif z_rule == "pyramid":
            return "Pyramid"
        elif z_rule == "hourglass":
            return "Hourglass_Polygon"
        elif z_rule == "rhombus":
            return "Rhombus_Polygon"
        else:
            raise ValueError()


def assign_classid(counter_class_):
    # Make default dict
    counter_class = [(i+1, xy_class, class2function(xy_class, z_class)) for i, (xy_class, z_class) in enumerate(counter_class_)]

    return counter_class, None, None


def random_rotate(box, target, rotate):
    if rotate == "stay":
        pass
    elif rotate == "side1":
        box = np.rot90(box, 1, axes=(0, 2))
        target = np.rot90(target, 1, axes=(0, 2))
    elif rotate == "side2":
        box = np.rot90(box, -1, axes=(0, 2))
        target = np.rot90(target, -1, axes=(0, 2))
    elif rotate == "side3":
        box = np.rot90(box, 1, axes=(1, 2))
        target = np.rot90(target, 1, axes=(1, 2))
    elif rotate == "side4":
        box = np.rot90(box, -1, axes=(1, 2))
        target = np.rot90(target, -1, axes=(1, 2))
    elif rotate == "upsidedown":
        box = np.rot90(box, 2, axes=(0, 2))
        target = np.rot90(target, 2, axes=(0, 2))

    return box, target


def random_setting(target, volume, filled_tensor):
    """
    Randomly define the position of primitive objects
    """
    conflict = True
    num = 0
    while conflict:
        filled_tensor_ = deepcopy(filled_tensor)
        target_ = deepcopy(target)
        tmp_tensor = np.zeros(SPACE)
        rotate = random.choice(ROTATE_L)
        _, target_ = random_rotate(target_, target_, rotate)
        w, h, d = target_.shape
        x = random.randint(0,SPACE[0]-w)
        y = random.randint(0,SPACE[1]-h)
        z = random.randint(0,SPACE[2]-d)
        tmp_tensor[x:x+w, y:y+h, z:z+d] = target_
        tmp_tensor = np.where(tmp_tensor>0, 1, 0)

        # Calculate overlap rate of volume 
        filled_tensor_ += tmp_tensor
        conflict_volume = np.sum(filled_tensor_>1)
        conflict_rate = conflict_volume / volume
        conflict = conflict_rate > args.occlusion_r
        #print("conflict_rate :", conflict_rate)
        
        if num > 100:
            #print("Unable to setting!")
            return None, filled_tensor, None
        num += 1

    #print("Setting succeeded!")

    # Remove overlapped region
    delete_tensor_mask = np.where(filled_tensor_>1, 0, 1)
    #print(np.sum(filled_tensor_>1))
    filled_tensor *= delete_tensor_mask

    return (x,y,z), filled_tensor, rotate


def setting_structure(coodinate, box, target, space, space_target, filled_tensor, rotate):
    tmp_space = np.zeros(SPACE, dtype=np.int64)
    tmp_space_target = np.zeros(SPACE, dtype=np.int64)
    box, target = random_rotate(box, target, rotate)
    x, y, z = coodinate
    w, h, d = target.shape

    #print(box.shape)
    #print(tmp_space[x:x+w, y:y+h, z:z+d].shape)

    # Set box in the 3D space (set pixel value 128)
    tmp_space[x:x+w, y:y+h, z:z+d] = box
    space += tmp_space
    space = np.where(space>0, 128, 0)

    # Set target mask in the 3D space
    #print(np.sum(space_target))
    space_target *= filled_tensor
    #print(np.sum(space_target))
    tmp_space_target[x:x+w, y:y+h, z:z+d] = target
    space_target += tmp_space_target

    # Update filled_tensor
    filled_tensor += tmp_space_target
    filled_tensor = np.where(filled_tensor>0, 1, 0)
  
    return space, space_target, filled_tensor


def make_data():
    boxes = []
    targets = []
    volumes = []

    # Instance generation
    for _ in range(NUM_PER_IMAGE):
        class_id, N, select_cont = random.choice(COUNTER_CLASS)
        f = getattr(C, select_cont)(N, class_id)
        box, target_box = f.construct_structure()
        boxes.append(box)
        targets.append(target_box)

        # Calculate volume
        volume = np.sum(target_box>0)
        volumes.append(volume)
        #print(volume)
    
    # Arranging the primitive instances into 3D volume 
    filled_tensor = np.zeros(SPACE)
    space = np.zeros(SPACE)
    space_target = np.zeros(SPACE)
    if args.noise_strength > 0:
        noise_level = random.randint(args.noise_range[0],args.noise_range[1])
        noise=np.random.randint(0, noise_level, SPACE)
    else:
        noise=np.zeros(SPACE, dtype=np.int64)

    indices = [i for i in range(len(volumes))]
    volumes, indices = zip(*sorted(zip(volumes, indices), reverse=True))

    for i in indices:
        volume = volumes[i]
        box = boxes[i]
        target = targets[i]
        coodinate, filled_tensor, rotate = random_setting(target, volume, filled_tensor)
        if coodinate is not None:
            space, space_target, filled_tensor = setting_structure(coodinate, box, target, space, space_target, filled_tensor, rotate)
    
    # Add noise
    space += noise  

    return space, space_target

def f(i):
    img,mask = make_data()

    # Save data
    img = img.astype(np.float64)
    mask = mask.astype(np.float64)
    img_nii=nib.Nifti1Image(img,np.eye(4))
    mask_nii=nib.Nifti1Image(mask,np.eye(4))
    nib.save(img_nii, "{}/img/img{:0=4}.nii.gz".format(args.save_dir, i))
    nib.save(mask_nii, "{}/label/label{:0=4}.nii.gz".format(args.save_dir, i))
    print(f"{i} complete!")
            

COUNTER_CLASS_ = list(itertools.product(XY_CLASS,Z_CLASS)) #+ [(0,"sphere")]
COUNTER_CLASS, IGNORE_XY_CONVERTER, IGNORE_Z_CONVERTER = assign_classid(COUNTER_CLASS_)
print(COUNTER_CLASS)
            
if __name__ == "__main__":
    mk_savedir()
    indices = [i for i in range(args.data_num) if i%WORKER_NUM==args.worker_id]
    print(f"total num : {len(indices)}")
    
    with ProcessPoolExecutor(10) as pool:
        for _ in pool.map(f, sorted(indices)):
            pass
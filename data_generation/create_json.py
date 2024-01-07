import os
import argparse
import glob
import json

parser = argparse.ArgumentParser()
parser.add_argument('--val_rate', type=float, default=0.1)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--base_json_name', type=str, default='dataset')
parser.add_argument('--label_name', type=str, default='label')
args = parser.parse_args()

def create_dict(args):
    img_paths = sorted(glob.glob(os.path.join(args.data_path, "img", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(args.data_path, args.label_name, "*.nii.gz")))
    assert len(img_paths) == len(label_paths)

    img_paths = sorted([o.replace(args.data_path, "") for o in img_paths])
    label_paths = sorted([o.replace(args.data_path, "") for o in label_paths])
    data_num = len(img_paths)
    print("data_num :", data_num)

    training_dicts = [{"image":im_path,"label":lb_path} for im_path,lb_path in zip(img_paths,label_paths)]
    val_num = int(len(img_paths)*args.val_rate)
    validation_dicts = [{"image":im_path,"label":lb_path} for im_path,lb_path in zip(img_paths[:val_num],label_paths[:val_num])]

    save_dict = {"training":training_dicts, "validation":validation_dicts}

    return save_dict, data_num

if __name__ == "__main__":
    save_dict, data_num = create_dict(args)
    json_name = f"{args.label_name}_num{data_num}.json"
    save_name = os.path.join(args.data_path,json_name)
    json_file1 = open(save_name, mode="w")
    json.dump(save_dict, json_file1)
    json_file1.close()
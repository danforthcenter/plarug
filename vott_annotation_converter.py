#!/usr/bin/env python

import os
import sys
import json
import glob
import random
import argparse
import shutil
import re


def options():
    parser = argparse.ArgumentParser(description="Convert VoTT annotaions to other formats.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", help="VoTT annotations export directory.", required=True)
    parser.add_argument("--outdir", help="Output directory.", required=True)
    parser.add_argument("--prop", help="Proportion of dataset to use for training.", default=0.8, type=float)
    #parser.add_argument("--format", help="Output format.", default="sagemaker")
    args = parser.parse_args()

    return args


def main():
    args = options()
    # Create the output directory if it does not exist
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    # Create the train subdirectory if it does not exist
    train_dir = os.path.join(args.outdir, "train")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    # Create the train_annotation subdirectory if it does not exist
    train_annotation_dir = os.path.join(args.outdir, "train_annotation")
    if not os.path.exists(train_annotation_dir):
        os.mkdir(train_annotation_dir)
    # Create the validation subdirectory if it does not exist
    val_dir = os.path.join(args.outdir, "validation")
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    # Create the validation_annotation subdirectory if it does not exist
    val_annotation_dir = os.path.join(args.outdir, "validation_annotation")
    if not os.path.exists(val_annotation_dir):
        os.mkdir(val_annotation_dir)

    # Read the VoTT annotation file
    vott_files = glob.glob(os.path.join(args.dir, "*.json"))
    if len(vott_files) > 1:
        # The VoTT format includes one JSON file with annotations for all images
        # If there is more than one file we do not know how to proceed
        print("Error: the VoTT annotation directory contains more than one JSON file, only one expected.",
              file=sys.stderr)
    with open(vott_files[0], "r") as fh:
        # Import the JSON annotation data
        vott = json.load(fh)

    # Get all the asset IDs and randomly shuffle the order
    asset_ids = list(vott["assets"].keys())
    random.shuffle(asset_ids)
    # Count how many assets are in the dataset
    asset_n = len(asset_ids)
    # Determine the number of assets in the training set
    train_n = int(args.prop * asset_n)
    # Split the asset IDs into a separate lists for train and val
    train_ids = asset_ids[:train_n]
    val_ids = asset_ids[train_n:]

    # Extract tags/categories/classes
    cat_id = 0
    categories = []
    cat_ids = {}
    for tag in vott["tags"]:
        category = {
            "class_id": cat_id,
            "name": tag["name"]
        }
        categories.append(category)
        cat_ids[tag["name"]] = cat_id
        cat_id += 1

    # Copy data and create annotation files
    reformat_data(asset_ids=train_ids, vott=vott, category_ids=cat_ids, categories=categories,
                  vott_dir=args.dir, img_dir=train_dir, annotation_dir=train_annotation_dir)
    reformat_data(asset_ids=val_ids, vott=vott, category_ids=cat_ids, categories=categories,
                  vott_dir=args.dir, img_dir=val_dir, annotation_dir=val_annotation_dir)


def reformat_data(asset_ids, vott, category_ids, categories, vott_dir, img_dir, annotation_dir):
    for asset_id in asset_ids:
        file_type = vott["assets"][asset_id]["asset"]["format"]
        filename = vott["assets"][asset_id]["asset"]["name"]
        prefix = re.sub("." + file_type, "", filename)
        meta = {
            "file": filename,
            "image_size": [
                {
                    "width": vott["assets"][asset_id]["asset"]["size"]["width"],
                    "height": vott["assets"][asset_id]["asset"]["size"]["height"],
                    "depth": 3
                }
            ],
            "annotations": [],
            "categories": categories
        }
        for region in vott["assets"][asset_id]["regions"]:
            annotation = {
                "class_id": category_ids[region["tags"][0]],
                "left": region["boundingBox"]["left"],
                "top": region["boundingBox"]["top"],
                "width": region["boundingBox"]["width"],
                "height": region["boundingBox"]["height"]
            }
            meta["annotations"].append(annotation)
        # Copy image file to image directory
        shutil.copyfile(os.path.join(vott_dir, filename), os.path.join(img_dir, filename))
        # Create annotation file
        anno_file = os.path.join(annotation_dir, prefix + ".json")
        with open(anno_file, "w") as f:
            json.dump(meta, f)


if __name__ == "__main__":
    main()

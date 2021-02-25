"use strict";

var _ = require('lodash');

const annotation_folder = "./annotations/";
const annotation_file = annotation_folder + "captions_train2014.json";

const image_folder = "./train2014/";

const fs = require("fs");

let annotations = JSON.parse(fs.readFileSync(annotation_file));

// Group all captions together having the same image ID.
const image_path_to_caption = {};

for (let val of annotations["annotations"]) {
  const caption = `<start> ${val["caption"]} <end>`;
  const image_path =
    image_folder + "COCO_train2014_" + "0".repeat(10) + val["image_id"] + ".jpg";
  if(!image_path_to_caption[image_path]) image_path_to_caption[image_path] = [];
  image_path_to_caption[image_path].push(caption);
}

const image_paths = Object.keys(image_path_to_caption);
image_paths = _.shuffle(image_paths);
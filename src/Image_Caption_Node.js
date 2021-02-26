"use strict";

var _ = require("lodash");

const annotation_folder = __dirname + "/annotations/";
const annotation_file = annotation_folder + "captions_train2014.json";

const image_folder = __dirname + "/train2014/";

const fs = require("fs");

const Server = require("./Server");

const tf = require("@tensorflow/tfjs-node");

let annotations = JSON.parse(fs.readFileSync(annotation_file));

// Group all captions together having the same image ID.
const image_path_to_caption = {};

for (let val of annotations["annotations"]) {
  const caption = `<start> ${val["caption"]} <end>`;
  const imgId = val["image_id"];
  const image_path =
    image_folder +
    "COCO_train2014_" +
    "0".repeat(12 - imgId.toString().length) +
    imgId +
    ".jpg";
  if (!image_path_to_caption[image_path])
    image_path_to_caption[image_path] = [];
  image_path_to_caption[image_path].push(caption);
}

let image_paths = Object.keys(image_path_to_caption);
image_paths = _.shuffle(image_paths);

// Select the first 6000 image_paths from the shuffled set.
// Approximately each image id has 5 captions associated with it, so that will
// lead to 30,000 examples.

const train_image_paths = image_paths.slice(0, 6000);

const train_captions = [],
  img_name_vector = [];

for (let image_path of train_image_paths) {
  const caption_list = image_path_to_caption[image_path];
  train_captions.push(caption_list);
  caption_list.forEach(() => {
    img_name_vector.push(image_path);
  });
}

// console.log(train_captions[0]);
// Server.showImage(img_name_vector[0]);

function load_image(image_path) {
  let img = fs.readFileSync(image_path);
  img = tf.node.decodeJpeg(img, 3);
  img = tf.image.resizeBilinear(img, [299, 299]);

  return { img, image_path };
}

const main = async () => {
  const { img, image_path } = await load_image(train_image_paths[0]);

  const image_model = await tf.loadLayersModel(
    "file://src/inception/model.json"
  );

  const new_input = image_model.input;
  const image_model_layers = image_model.layers;
  const hidden_layer = image_model_layers[image_model_layers.length - 1].output;

  const image_features_extract_model = tf.model({
    inputs: new_input,
    outputs: hidden_layer,
  });

  // Get unique images
  const encode_train = [...new Set(img_name_vector)].sort((a, b) => a - b);

  // Feel free to change batch_size according to your system configuration
  let image_dataset = encode_train.map(load_image);
  image_dataset = tf.data.array(image_dataset).batch(16);

  await image_dataset.forEachAsync((imgData) => {
    const { img, image_path } = imgData;
    console.log({ img, image_path });
    return
  });
};

main();

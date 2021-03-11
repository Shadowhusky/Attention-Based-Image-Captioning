"use strict";

const _ = require("lodash");

const annotation_folder = __dirname + "/annotations/";
const annotation_file = annotation_folder + "captions_train2014.json";

const image_folder = __dirname + "/train2014/";

const fs = require("fs");

const Server = require("./Server");

const tf = require("@tensorflow/tfjs-node");

const { Tokenizer } = require("tf_node_tokenizer");

const iniModel = require("./model/models");

const initialize = () => {
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

  const train_image_paths = image_paths.slice(0, 8000);

  let train_captions = [],
    img_name_vector = [];

  for (let image_path of train_image_paths) {
    const caption_list = image_path_to_caption[image_path];
    train_captions = train_captions.concat(caption_list);
    caption_list.forEach(() => {
      img_name_vector.push(image_path);
    });
  }

  return { train_captions, img_name_vector };
};

// console.log(train_captions[0]);
// Server.showImage(img_name_vector[0]);

function load_image(image_path) {
  let img = fs.readFileSync(image_path);
  img = tf.node.decodeJpeg(img, 3);
  img = tf.image.resizeBilinear(img, [299, 299]);

  return { img, image_path };
}

const imgData_path = __dirname + "/temp/imgData.json";

const initialized = fs.existsSync(imgData_path);

const main = async () => {
  let train_captions, img_name_vector;
  if (initialized) {
    let imgData = fs.readFileSync(imgData_path);
    imgData = JSON.parse(imgData);
    train_captions = imgData.train_captions;
    img_name_vector = imgData.img_name_vector;
  } else {
    const imgData = initialize();
    train_captions = imgData.train_captions;
    img_name_vector = imgData.img_name_vector;
    // Save data to improve performance
    fs.writeFileSync(imgData_path, JSON.stringify(imgData), "utf-8");
  }

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

  const extractAndSaveFeatureVector = async () => {
    // Feel free to change batch_size according to your system configuration
    let image_dataset;

    image_dataset = encode_train.map(load_image);
    image_dataset = tf.data.array(image_dataset).batch(16);

    await image_dataset.forEachAsync((imgData) => {
      const { img, image_path } = imgData;
      let batch_features = image_features_extract_model.predict(img);
      batch_features = batch_features.reshape([
        batch_features.shape[0],
        -1,
        batch_features.shape[3],
      ]);
      for (let { 0: bf, 1: p } of _.zip(
        batch_features.arraySync(),
        image_path.arraySync()
      )) {
        fs.writeFileSync(
          p + ".temp",
          JSON.stringify({
            data: bf,
            shape: batch_features.shape,
          }),
          "utf-8"
        );
      }
    });
  };

  if (!initialized) await extractAndSaveFeatureVector();

  const calc_max_length = (seqs) => {
    return _.max(seqs.map((seq) => seq.length));
  };

  // Pad each vector to the maxLength of the captions
  const padSequences = (seqs, maxLength) => {
    maxLength = maxLength ? maxLength : calc_max_length(seqs);
    const padSeq = (seq) => {
      const padding = new Array(maxLength - seq.length).fill(0);
      return seq.concat(padding);
    };
    return seqs.map((seq) => padSeq(seq));
  };

  const top_k = 5000;
  const tokenizer = new Tokenizer({
    num_words: top_k,
    oov_token: "<ukn>",
    filters: '!"#$%&()*+.,-/:;=?@[]^_`{|}~ ',
  });
  tokenizer.fitOnTexts(train_captions);
  tokenizer.word_index["<pad>"] = 0;
  tokenizer.index_word[0] = "<pad>";

  const train_seqs = tokenizer.textsToSequences(train_captions);
  const max_length = calc_max_length(train_seqs);
  const cap_vector = padSequences(train_seqs, max_length);

  // Split the data into training and testing
  const img_to_cap_vector = {};
  for (let { 0: img, 1: cap } of _.zip(img_name_vector, cap_vector)) {
    if (!img_to_cap_vector[img]) img_to_cap_vector[img] = [];
    img_to_cap_vector[img].push(cap);
  }

  // Create training and validation sets using an 80-20 split randomly.
  let img_keys = Object.keys(img_to_cap_vector);
  img_keys = _.shuffle(img_keys);

  const slice_index = parseInt(img_keys.length * 0.8);
  const img_name_train_keys = img_keys.slice(0, slice_index);
  const img_name_val_keys = img_keys.slice(slice_index);

  let img_name_train = [];
  let cap_train = [];
  for (let imgt of img_name_train_keys) {
    img_to_cap_vector[imgt].forEach(() => {
      img_name_train = img_name_train.concat(imgt);
    });
    cap_train = cap_train.concat(img_to_cap_vector[imgt]);
  }
  let img_name_val = [];
  let cap_val = [];
  for (let imgv of img_name_val_keys) {
    img_to_cap_vector[imgv].forEach(() => {
      img_name_val = img_name_val.concat(imgv);
    });
    cap_val = cap_val.concat(img_to_cap_vector[imgv]);
  }

  // Training parameters
  const BATCH_SIZE = 64;
  const BUFFER_SIZE = 1000;
  const embedding_dim = 256;
  const units = 512;
  const vocab_size = top_k + 1;
  const num_steps = img_name_train.length;
  // Shape of the vector extracted from InceptionV3 is (64, 2048)
  // These two variables represent that vector shape
  const features_shape = 2048;
  const attention_features_shape = 64;

  let dataset = tf.data.array(_.zip(img_name_train, cap_train));

  // Use map to load the cached bf files in parallel
  dataset = dataset.map(({ 0: img, 1: cap }) => {
    return load_batch_features(img, cap);
  });

  // Shuffle and batch
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE);
  // dataset = dataset.prefetch(6);

  // Train the model
  await trainModel(dataset, tokenizer, {
    num_steps,
    embedding_dim,
    units,
    vocab_size,
  });
};

const load_batch_features = (img_name, cap) => {
  let temp = JSON.parse(fs.readFileSync(img_name + ".temp"));
  return {
    img_tensor: tf.tensor(temp.data, [temp.shape[1], temp.shape[2]]),
    cap,
  };
};

main();

const trainModel = async (dataset, tokenizer, options) => {
  const { num_steps, embedding_dim, units, vocab_size } = options;
  const EPOCHS = 2;

  const Model = new iniModel(embedding_dim, units, vocab_size);
  
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const start = new Date().getMilliseconds();
    let total_loss = tf.tensor(0);

    let batch = 0;

    await dataset.forEachAsync(({ img_tensor, cap: target }) => {
      const { loss: batch_loss, total_loss: t_loss } = Model.train_step(
        img_tensor,
        target,
        tokenizer
      );
      total_loss = total_loss.add(t_loss);

      if (batch++ % 5 === 0) {
        const lossData = batch_loss.dataSync()[0];
        console.log(
          `Epoch ${epoch + 1} Batch ${batch} Loss ${
            lossData / parseInt(target.shape[1])
          }`
        );
      }
    });

    if (epoch % 5 === 0) {
      // Save model
    }

    console.log(
      `Epoch ${epoch + 1} Loss ${
        total_loss.div(tf.tensor(num_steps)).dataSync()[0]
      }`
    );
    console.log(
      t`Time taken for 1 epoch ${
        (new Date().getMilliseconds() - start) / 1000
      } sec`
    );
  }
};

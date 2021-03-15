"use strict";

const _ = require("lodash");

const annotation_folder = __dirname + "/annotations/";
const annotation_file = annotation_folder + "captions_train2014.json";

const image_folder = __dirname + "/train2014/";

const fs = require("fs");

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

  // Select the first 8000 image_paths from the shuffled set.
  // Approximately each image id has 5 captions associated with it, so that will
  // lead to 40,000 examples.

  const train_image_paths = image_paths.slice(0, 5000);

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

function load_image(image_path) {
  let img = fs.readFileSync(image_path);
  img = tf.node.decodeJpeg(img, 3);
  img = tf.image.resizeBilinear(img, [299, 299]);
  // Preprocess the images using the preprocess_input method to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.
  // The formular was referenced from the _preprocess_numpy_input method in this github file:
  // https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/applications/imagenet_utils.py#L169
  // mode == 'tf'
  img = img.div(tf.scalar(127.5)).sub(tf.scalar(1));
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
  // const hidden_layer = image_model_layers[image_model_layers.length - 1].output;
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
            shape: batch_features.shape.slice(1),
          }),
          "utf-8"
        );
      }
    });
  };

  if (!initialized) {
    await extractAndSaveFeatureVector();
  }

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
    filters: /[\\.,/#!$%^&*;?"'@:{}=\-_`~()]/g,
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
  const BUFFER_SIZE = 1500;
  const embedding_dim = 256;
  const units = 512;
  const vocab_size = Object.keys(tokenizer.word_index).length;
  const num_steps = parseInt(img_name_train.length / BATCH_SIZE);
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
  dataset = dataset.prefetch(64);

  // Train the model
  await trainModel(
    dataset,
    tokenizer,
    {
      num_steps,
      embedding_dim,
      units,
      vocab_size,
    },
    image_features_extract_model,
    {
      image_features_extract_model,
      tokenizer,
      max_length,
      img_name_val,
      cap_val,
    }
  );
};

const load_batch_features = (img_name, cap) => {
  let temp = JSON.parse(fs.readFileSync(img_name + ".temp"));
  return {
    img_tensor: tf.tensor(temp.data, temp.shape),
    cap,
  };
};

const saved_model_path = __dirname + "/model/saved_model.json";

const saveModel = (trainable_variables) => {
  try {
    if (!trainable_variables || !trainable_variables[0].name_) {
      return;
    }
    fs.writeFileSync(
      saved_model_path,
      JSON.stringify(
        trainable_variables.map((variable) => {
          return {
            data: variable.arraySync(),
            shape: variable.shape,
            name_: variable.name_,
          };
        })
      ),
      "utf-8"
    );
  } catch (err) {
    console.error("saveModel ", err);
  }
};

const trainModel = async (
  dataset,
  tokenizer,
  options,
  image_features_extract_model,
  evaConfig
) => {
  const { num_steps, embedding_dim, units, vocab_size } = options;
  const EPOCHS = 10;

  let Model;

  try {
    const trainable_variables = JSON.parse(
      fs.readFileSync(saved_model_path)
    ).map((variable) => {
      const variable_ = tf.tensor(variable.data, variable.shape);
      variable_.name_ = variable.name_;
      return variable_;
    });
    Model = new iniModel(embedding_dim, units, vocab_size, trainable_variables);
  } catch (err) {
    Model = new iniModel(embedding_dim, units, vocab_size);
  }

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const start = new Date().getTime();
    let total_loss = tf.scalar(0);

    let batch = 0;

    await dataset.forEachAsync(({ img_tensor, cap: target }) => {
      if (img_tensor.shape[0] === 64) {
        // Executes the provided function fn and after it is executed, cleans up all intermediate tensors allocated by fn except those returned by fn.
        const {
          loss: batch_loss,
          total_loss: t_loss,
          trainable_variables,
        } = tf.tidy(() => {
          return Model.train_step(img_tensor, target, tokenizer);
        });
        total_loss = total_loss.add(t_loss);
        if (batch++ % 5 === 0) {
          console.log("Number of tensors: " + tf.memory().numTensors);
          const lossData = batch_loss.dataSync()[0];
          console.log(
            `Epoch ${epoch} Batch ${batch} Loss ${
              lossData / parseInt(target.shape[1])
            }`
          );
          testOnImg({ Model, ...evaConfig });
          saveModel(trainable_variables);
        }
      }
    });
    
    console.log(
      `Epoch ${epoch + 1} Loss ${
        total_loss.div(tf.scalar(num_steps)).dataSync()[0]
      }`
    );
    console.log(
      `Time taken for epoch ${epoch + 1}: ${
        (new Date().getTime() - start) / 1000
      } sec`
    );
  }
};

function evaluate(
  image,
  Model,
  image_features_extract_model,
  tokenizer,
  max_length
) {
  return tf.tidy(() => {
    const { encoder, decoder } = Model;
    let hidden = decoder.reset_state(1);

    const temp_input = tf.expandDims(load_image(image).img, 0);
    let img_tensor_val = image_features_extract_model.predict(temp_input);
    img_tensor_val = tf.reshape(img_tensor_val, [
      img_tensor_val.shape[0],
      -1,
      img_tensor_val.shape[3],
    ]);

    const features = encoder.call(img_tensor_val);

    let dec_input = tf.expandDims([tokenizer.word_index["<start>"]], 0);
    const result = [];

    for (let i = 0; i < max_length; i++) {
      const { predictions, hidden: hidden_, attention_weights } = decoder.call(
        dec_input,
        features,
        hidden
      );
      hidden = hidden_;

      const predicted_id = tf.multinomial(predictions, 1).arraySync()[0][0];
      result.push(tokenizer.index_word[predicted_id]);

      if (tokenizer.index_word[predicted_id] === "<end>") {
        return result;
      }

      dec_input = tf.expandDims([predicted_id], 0);
    }
    return result;
  });
}

function testOnImg(configs) {
  const {
    Model,
    image_features_extract_model,
    tokenizer,
    max_length,
    img_name_val,
    cap_val,
  } = configs;
  const rid = _.random(0, img_name_val.length);
  const image = img_name_val[rid];
  const result = evaluate(
    image,
    Model,
    image_features_extract_model,
    tokenizer,
    max_length
  );

  const real_caption = parseCap(cap_val[rid], tokenizer);

  console.log("Image Path: ", img_name_val[rid]);
  console.log("Real Caption: ", real_caption);
  console.log("Prediction Caption: " + result.join(" "));
}

const parseCap = (capIndecies, tokenizer) => {
  return capIndecies
    .filter((cap) => cap !== 0)
    .map((cap) => {
      return tokenizer.index_word[cap];
    })
    .join(" ");
};

main();

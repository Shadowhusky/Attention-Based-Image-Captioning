const { onesLike } = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node");

const _ = require("lodash");

function updateTrainingWeights(...layers) {
  if (this.updatedTrainingWeights) return;
  else this.updatedTrainingWeights = true;
  for (layer of layers) {
    if (!layer.trainableWeights) continue;
    this.trainableWeights_ = this.trainableWeights_.concat(
      layer.trainableWeights.map((_) => {
        const variable = _.val;
        variable.name_ = layer.name;
        return variable;
      })
    );
  }
}

class BahdanauAttention {
  constructor(units, trainable_variables) {
    this.trainableWeights_ = [];
    this.W1 = tf.layers.dense({
      units,
      name: "bahdanau_dense1",
      weights: trainable_variables
        ? trainable_variables["bahdanau_dense1"]
        : undefined,
    });
    this.W2 = tf.layers.dense({
      units,
      name: "bahdanau_dense2",
      weights: trainable_variables
        ? trainable_variables["bahdanau_dense2"]
        : undefined,
    });
    this.V = tf.layers.dense({
      units: 1,
      name: "bahdanau_dense3",
      weights: trainable_variables
        ? trainable_variables["bahdanau_dense3"]
        : undefined,
    });
  }

  call(features, hidden) {
    // features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    // hidden shape == (batch_size, hidden_size)
    // hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    const hidden_with_time_axis = hidden.expandDims(1);

    // attention_hidden_layer shape == (batch_size, 64, units)
    const attention_hidden_layer = tf.tanh(
      tf.add(this.W1.apply(features), this.W2.apply(hidden_with_time_axis))
    );

    // score shape == (batch_size, 64, 1)
    // This gives you an unnormalized score for each image feature.
    const score = this.V.apply(attention_hidden_layer);

    // attention_weights shape == (batch_size, 64, 1)
    const attention_weights = score.softmax();

    // context_vector shape after sum == (batch_size, hidden_size)
    let context_vector = tf.mul(attention_weights, features);
    context_vector = context_vector.sum(1);

    const bahdanauLayers = [this.W1, this.W2, this.V];

    return { context_vector, attention_weights, bahdanauLayers };
  }
}

class CNN_Encoder {
  // Since you have already extracted the features and dumped it using pickle
  // This encoder passes those features through a Fully connected layer

  constructor(embedding_dim, trainable_variables) {
    // shape after fc == (batch_size, 64, embedding_dim
    this.trainableWeights_ = [];
    this.fc = tf.layers.dense({
      units: embedding_dim,
      name: "encoder_dense",
      weights: trainable_variables
        ? trainable_variables["encoder_dense"]
        : undefined,
    });
  }

  call(x) {
    x = this.fc.apply(x);
    x = x.relu();
    updateTrainingWeights.apply(this, [this.fc]);
    return x;
  }
}

class RNN_Decoder {
  constructor(embedding_dim, units, vocab_size, trainable_variables) {
    this.units = units;
    this.trainableWeights_ = [];
    this.embedding = tf.layers.embedding({
      inputDim: vocab_size,
      outputDim: embedding_dim,
      name: "decoder_embedding",
      weights: trainable_variables
        ? trainable_variables["decoder_embedding"]
        : undefined,
    });
    this.gru = tf.layers.gru({
      units: units,
      returnSequences: true,
      returnState: true,
      recurrentInitializer: "glorotUniform",
      name: "decoder_gru",
      weights: trainable_variables
        ? trainable_variables["decoder_gru"]
        : undefined,
    });
    this.fc1 = tf.layers.dense({
      units: units,
      name: "decoder_dense1",
      weights: trainable_variables
        ? trainable_variables["decoder_dense1"]
        : undefined,
    });
    this.fc2 = tf.layers.dense({
      units: vocab_size,
      name: "decoder_dense2",
      weights: trainable_variables
        ? trainable_variables["decoder_dense2"]
        : undefined,
    });

    this.attention = new BahdanauAttention(units, trainable_variables);
  }

  call(x, features, hidden) {
    // defining attention as a separate model
    const {
      context_vector,
      attention_weights,
      bahdanauLayers,
    } = this.attention.call(features, hidden);

    // x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = this.embedding.apply(x);

    // x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([context_vector.expandDims(1), x], x.rank - 1);

    // Have to do this hack because of problems with rnn layer
    // Check https://github.com/tensorflow/tfjs/issues/3550
    x = x.concat(x, 1);

    // passing the concatenated vector to the GRU

    let [output, state] = this.gru.apply(x);
    // shape == (batch_size, max_length, hidden_size)

    x = this.fc1.apply(output);

    // Restore the shape changed by the hack mentioned above
    x = x.slice([0, 1], [-1]);

    // x shape == (batch_size * max_length, hidden_size)
    x = x.reshape([-1, x.shape[2]]);

    // output shape == (batch_size * max_length, vocab)
    x = this.fc2.apply(x);

    updateTrainingWeights.apply(this, [
      this.fc1,
      this.fc2,
      this.attention,
      this.gru,
      this.embedding,
      ...bahdanauLayers,
    ]);

    return {
      predictions: x,
      hidden: state,
      attention_weights,
    };
  }

  reset_state(batch_size) {
    return tf.zeros([batch_size, this.units]);
  }
}

const loss_object = (labels, pred) => {
  return tf.losses.softmaxCrossEntropy(labels, pred, undefined, undefined, 0);
};

function loss_function(labels_onehot, pred, labels_index) {
  let mask = labels_index.equal(0).logicalNot();
  let loss_ = loss_object(labels_onehot, pred);

  mask = tf.cast(mask, loss_.dtype);
  loss_ = loss_.mul(mask);

  return loss_.mean();
}

class iniModel {
  constructor(embedding_dim, units, vocab_size, trainable_variables) {
    if (trainable_variables) {
      this.trainable_variables = {};
      trainable_variables.forEach((variable) => {
        let nameWeightsMap = this.trainable_variables[variable.name_];
        if (nameWeightsMap) {
          nameWeightsMap.push(variable);
        } else {
          this.trainable_variables[variable.name_] = [variable];
        }
      });
    }

    this.encoder = new CNN_Encoder(embedding_dim, this.trainable_variables);
    this.decoder = new RNN_Decoder(
      embedding_dim,
      units,
      vocab_size,
      this.trainable_variables
    );
    this.units = units;
    this.optimizer = tf.train.adam();
  }

  train_step(img_tensor, target, tokenizer) {
    const trainable_variables_saved = this.trainable_variables
      ? this.trainable_variables
      : null;
    const decoder = this.decoder;
    const encoder = this.encoder;

    // initializing the hidden state for each batch
    // because the captions are not related from image to image
    const hidden = decoder.reset_state(target.shape[0]);

    let dec_input = tf.expandDims(
      Array(target.shape[0]).fill(tokenizer.word_index["<start>"]),
      1
    );

    const lossFunc = (target, dec_input, hidden) => {
      const features = encoder.call(img_tensor);
      let loss = tf.tensor(0);
      for (let i = 1; i < target.shape[1]; i++) {
        // passing the features through the decoder
        const { predictions, hidden: hidden_ } = decoder.call(
          dec_input,
          features,
          hidden
        );
        hidden = hidden_;

        const currTarget = target
          .slice([0, i], [-1, 1])
          .squeeze()
          .cast("int32");

        let target_one_hot = currTarget;

        target_one_hot = target_one_hot.oneHot(predictions.shape[1]);

        loss = loss.add(loss_function(target_one_hot, predictions, currTarget));

        // using teacher forcing
        dec_input = currTarget.expandDims(1);
      }
      // loss = attachWeight(loss);
      return loss;
    };

    let loss = lossFunc(target, dec_input, hidden);

    const trainable_variables = decoder.trainableWeights_.concat(
      encoder.trainableWeights_
    );

    const { grads } = tf.variableGrads(
      () => lossFunc(target, dec_input, hidden),
      trainable_variables
    );

    const total_loss = loss.div(tf.tensor(target.shape[1]));

    this.optimizer.applyGradients(grads);

    return { loss, total_loss, trainable_variables };
  }
}

module.exports = iniModel;

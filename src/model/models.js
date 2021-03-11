const { onesLike } = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node");

const _ = require("lodash");

function updateTrainingWeights(...layers) {
  if (this.updatedTrainingWeights) return;
  else this.updatedTrainingWeights = true;
  for (layer of layers) {
    if (!layer.trainableWeights) continue;
    this.trainableWeights = this.trainableWeights.concat(
      layer.trainableWeights.map((_) => _.val)
    );
  }
}

class BahdanauAttention {
  constructor(units) {
    this.trainableWeights = [];
    this.W1 = tf.layers.dense({ units });
    this.W2 = tf.layers.dense({ units });
    this.V = tf.layers.dense({ units: 1 });
    return (features, hidden) => {
      return this.call(features, hidden);
    };
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

    updateTrainingWeights.apply(this, [this.W1, this.W2, this.V]);

    return { context_vector, attention_weights };
  }
}

class CNN_Encoder {
  // Since you have already extracted the features and dumped it using pickle
  // This encoder passes those features through a Fully connected layer

  constructor(embedding_dim) {
    // shape after fc == (batch_size, 64, embedding_dim
    this.trainableWeights = [];
    this.fc = tf.layers.dense({ units: embedding_dim });
  }

  call(x) {
    x = this.fc.apply(x);
    x = x.relu();
    updateTrainingWeights.apply(this, [this.fc]);
    return x;
  }
}

class RNN_Decoder {
  constructor(embedding_dim, units, vocab_size) {
    this.units = units;
    this.trainableWeights = [];
    this.embedding = tf.layers.embedding({
      inputDim: vocab_size,
      outputDim: embedding_dim,
    });
    this.gru = tf.layers.gru({
      units: units,
      returnSequences: true,
      returnState: true,
      recurrentInitializer: "glorotUniform",
    });
    this.fc1 = tf.layers.dense({ units: units });
    this.fc2 = tf.layers.dense({ units: vocab_size });

    this.attention = new BahdanauAttention(units);
  }

  call(x, features, hidden) {
    // defining attention as a separate model
    const { context_vector, attention_weights } = this.attention(
      features,
      hidden
    );

    // x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = this.embedding.apply(x);

    // x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([context_vector.expandDims(1), x], x.rank - 1);

    // passing the concatenated vector to the GRU
    let [output, state] = this.gru.apply(x);
    // shape == (batch_size, max_length, hidden_size)

    // Have to do this hack because of problems with rnn layer
    // Check https://github.com/tensorflow/tfjs/issues/3550
    output = tf.tensor(output.dataSync(), output.shape);
    state = tf.tensor(state.dataSync(), state.shape);

    x = this.fc1.apply(output);

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
  constructor(embedding_dim, units, vocab_size) {
    this.encoder = new CNN_Encoder(embedding_dim);
    this.decoder = new RNN_Decoder(embedding_dim, units, vocab_size);
    this.units = units;
    this.optimizer = tf.train.adam();
  }

  train_step(img_tensor, target, tokenizer) {
    const decoder = this.decoder;
    const encoder = this.encoder;

    // initializing the hidden state for each batch
    // because the captions are not related from image to image
    const hidden = decoder.reset_state(target.shape[0]);

    let dec_input = tf.expandDims(
      Array(target.shape[0]).fill(tokenizer.word_index["<start>"]),
      1
    );

    const attachWeight = (loss) => {
      for (let weights of decoder.trainableWeights.concat(
        encoder.trainableWeights
      )) {
        loss = weights.sum().mul(tf.tensor(0)).add(loss);
      }
      return loss;
    };

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
      loss = attachWeight(loss);
      return loss;
    };

    let loss = lossFunc(target, dec_input, hidden);

    const trainable_variables = decoder.trainableWeights.concat(
      encoder.trainableWeights
    );

    const { grads } = tf.variableGrads(
      () => lossFunc(target, dec_input, hidden),
      trainable_variables
    );

    const total_loss = loss.div(tf.tensor(target.shape[1]));

    this.optimizer.applyGradients(grads);

    return { loss, total_loss };
  }
}

module.exports = iniModel;

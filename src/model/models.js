const tf = require("@tensorflow/tfjs-node");

class BahdanauAttention {
  constructor(units) {
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
    context_vector = context_vector.sum();

    return { context_vector, attention_weights };
  }
}

class CNN_Encoder {
  // Since you have already extracted the features and dumped it using pickle
  // This encoder passes those features through a Fully connected layer

  constructor(embedding_dim) {
    // shape after fc == (batch_size, 64, embedding_dim)
    this.fc = tf.layers.dense({ units: embedding_dim });
  }

  call(x) {
    x = this.fc.apply(x);
    x = x.relu();
    return x;
  }
}

class RNN_Decoder {
  constructor(embedding_dim, units, vocab_size) {
    this.units = units;

    this.embedding = tf.layers.embedding({
      inputDim: vocab_size,
      outputDim: embedding_dim,
    });
    this.gru = tf.layers.gru({
      units: this.units,
      returnSequences: true,
      returnState: true,
      recurrentInitializer: "glorotUniform",
    });
    this.fc1 = tf.layers.dense({ units: this.units });
    this.fc2 = tf.layers.dense({ units: vocab_size });

    this.attention = new BahdanauAttention(this.units);
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
    const state = this.gru.apply(x);
    const output = state;
    // shape == (batch_size, max_length, hidden_size)
    x = this.fc1.apply(output);

    // x shape == (batch_size * max_length, hidden_size)
    x = x.reshape([-1, x.shape[2]]);

    // output shape == (batch_size * max_length, vocab)
    x = this.fc2.apply(x);

    return { x, state, attention_weights };
  }

  reset_state(batch_size) {
    return tf.zeros([batch_size, this.units]);
  }
}

const optimizer = tf.train.adam();
const loss_object = (labels, pred) => {
  return tf.losses.meanSquaredError(labels, pred, undefined, 0);
};

function loss_function(labels, pred) {
  let mask = labels.equal(0).logicalNot();
  let loss_ = loss_object(labels, pred);

  mask = tf.cast(mask, loss_.dtype);
  loss_ = loss_.mul(mask);

  return loss_.mean();
}
 
# Text-Summarization-With-Amazon-Reviews
A text-summarization sequence-to-sequence model with an attention mechanism. 

The model was trained on a subset of the  [Amazon-Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset, which includes over 500,000 product reviews, and utilized [GloVe](https://www.kaggle.com/datasets/watts2/glove6b50dtxt) embeddings for word representation.

## Architecture
A seq2seq model consisting of an encoder, attention mechanism, and a decoder inspired by the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). 

#### Encoder Layer
- An embedding layer followed by a 2-layer Bi-directional LSTM. 

- The resulting output is the bi-directional LSTM sequence output and the final concatenated forward and backward hidden and memory states.

#### Attention Layer
- The previous decoder's hidden state is replicated Tx times, where Tx represents the length of the source sequence. 

- This replicated state is then concatenated with the encoder's output and passed through two fully connected layers. A softmax function is applied to produce attention scores for each token in the sequence. 

- These attention scores are then used in a dot product operation with the encoder outputs to generate a context vector for that specific timestep.

#### Decoder Layer
- An embedding layer is applied to the target/predicted token and concatenated with the context vector obtained from the attention layer.

- This combined input is then passed through a 3-layer LSTM, which is initialized using the concatenated hidden and memory states from the encoder.


## Training
- The source sequence is passed through the encoder layer, which produces the sequence outputs for each timestep, along with the final hidden and memory states.

- Initially, the SOS token, along with the 3 outputs from the encoder, is provided as input to the decoder layer.

- Subsequently, the token fed into the decoder is selected stochastically based on a teacher forcing ratio, which governs whether the true target token or the predicted token from the previous timestep is used as input during decoding.


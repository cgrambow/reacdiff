import keras

import reacdiff.models.conv as conv


class CRNN:
    """A recurrent neural network with convolutional (DenseNet) encoding"""

    def __init__(self,
                 input_shape=(100, 128, 128, 1),
                 output_dim=26,
                 observables=1,
                 rnn_layers=1,
                 rnn_units=100,
                 use_gpu=True):
        """
        :param input_shape: shape of input tensor.
        :param output_dim: int, number of output units.
        :param observables: int, number of observables.
        :param rnn_layers: int, number of RNN layers
        :param rnn_units: int, number of units in RNN
        :param use_gpu: bool, use GPU.
        """
        self.input_shape = input_shape
        self.inputs = [keras.layers.Input(shape=input_shape, name=f'input_{i+1}') for i in range(observables)]

        self.encoder_outputs = None
        self.rnn = None
        self.model = None

        self.output_dim = output_dim
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.use_gpu = use_gpu

    def build_rnn(self, **kwargs):
        """
        :param kwargs: DenseNet args
        """
        inputs = keras.layers.Input(shape=self.input_shape, name='conv_input')

        # Build DenseNet encoder
        dense_net = conv.DenseNet(inputs=inputs, **kwargs)
        dense_net.build()
        self.encoder_outputs = dense_net.outputs
        seqs = self.encoder_outputs

        # RNN
        if self.rnn_layers > 1:
            rnn_layers = [seqs]
            for i in range(self.rnn_layers - 1):
                prev_layer = rnn_layers[-1]
                rnn_layers.append(rnn(self.rnn_units,
                                      return_sequences=True,
                                      gpu=self.use_gpu,
                                      name='rnn' + str(i + 1))(prev_layer))

            # Use skip connections between RNN layers if using multiple RNN layers
            seqs = keras.layers.Concatenate(name='rnn_concat')(rnn_layers)

        rnn_outputs = rnn(self.rnn_units,
                          return_sequences=False,
                          gpu=self.use_gpu,
                          name='rnn_last')(seqs)

        self.rnn = keras.models.Model(inputs, rnn_outputs, name='rnn')

    def build(self, **kwargs):
        """
        :param kwargs: DenseNet args
        """
        # Encode each observable with the same RNN
        self.build_rnn(**kwargs)
        rnn_outputs = [self.rnn(tensor) for tensor in self.inputs]

        # Output layer
        if len(rnn_outputs) > 1:
            rnn_outputs = keras.layers.Concatenate(name='output_concat')(rnn_outputs)
        else:
            rnn_outputs = rnn_outputs[0]
        outputs = keras.layers.Dense(self.output_dim, name='output')(rnn_outputs)

        self.model = keras.models.Model(self.inputs, outputs, name='crnn')


def rnn(units, return_sequences=False, gpu=True, name=''):
    if gpu:
        return keras.layers.CuDNNLSTM(units,
                                      return_sequences=return_sequences,
                                      name=name)
    else:
        # Make recurrent_activation compatible with CuDNNLSTM
        return keras.layers.LSTM(units,
                                 return_sequences=return_sequences,
                                 recurrent_activation='sigmoid',
                                 name=name)

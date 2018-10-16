'''Author: Brandon Trabucco, Copyright 2019
Implements the bottom-up top-down visual attention LSTM cell for image captioning.
Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''


import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
import collections


def _softmax_attention(x):
    # x is shaped: [batch, num_boxes, depth]
    x = tf.transpose(x, [0, 2, 1])
    return tf.transpose(tf.nn.softmax(x), [0, 2, 1])


def _sigmoid_attention(x):
    # x is shaped: [batch, num_boxes, depth]
    x_size = tf.to_float(tf.shape(x)[1])
    return tf.nn.sigmoid(x) / x_size


def _tile_new_axis(x, n, d):
    # expand and tile new dimensions of x
    nd = zip(n, d)
    nd = sorted(nd, key=lambda ab: ab[1])
    n, d = zip(*nd)
    for i in sorted(d):
        x = tf.expand_dims(x, i)
    reverse_d = {val: idx for idx, val in enumerate(d)}
    tiles = [n[reverse_d[i]] if i in d else 1 for i, _s in enumerate(x.shape)]
    return tf.tile(x, tiles)


# Used to store the internal states of each LSTM.
_UpDownStateTuple = collections.namedtuple("UpDownStateTuple", ("v", "l"))


# Wrapper for _UpDownStateTuple.
class UpDownStateTuple(_UpDownStateTuple):
    """Tuple used by UpDown Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(v, l)`, in that order.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (v, l) = self
        if not v.dtype == l.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(v.dtype), str(l.dtype)))
        return v.dtype


# The wrapper for the up-down attention mechanism
class UpDownCell(tf.contrib.rnn.LayerRNNCell):
    '''Implements the bottom-up top-down attention mechanism from
    Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''

    def __init__(self, 
            # The default LSTM parameters
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name=None, dtype=None,
            # The extra parameters for the up-down mechanism
            mean_global_features=None, mean_object_features=None,
            attention_method='softmax', **kwargs ):
        """Initialize the parameters for an LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            use_peepholes: bool, set True to enable diagonal/peephole connections.
            cell_clip: (optional) A float value, if provided the cell state is clipped
                by this value prior to the cell output activation.
            initializer: (optional) The initializer to use for the weight and
                projection matrices.
            num_proj: (optional) int, The output dimensionality for the projection
                matrices.  If None, no projection is performed.
            proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
                provided, then the projected values are clipped elementwise to within
                `[-proj_clip, proj_clip]`.
            num_unit_shards: Deprecated, will be removed by Jan. 2017.
                Use a variable_scope partitioner instead.
            num_proj_shards: Deprecated, will be removed by Jan. 2017.
                Use a variable_scope partitioner instead.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of
                the training. Must set it manually to `0.0` when restoring from
                CudnnLSTM trained checkpoints.
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.  If False, they are concatenated
                along the column axis.  This latter behavior will soon be deprecated.
            activation: Activation function of the inner states.  Default: `tanh`. It
                could also be string that is within Keras activation function names.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already has
                the given variables, an error is raised.
            name: String, the name of the layer. Layers with the same name will
                share weights, but to avoid mistakes we require reuse=True in such
                cases.
            dtype: Default dtype of the layer (default of `None` means use the type
                of the first input). Required when `build` is called before `call`.
            mean_global_features: float32 Tensor, average pooled image features, 
                with shape [batch_size, feature_depth] (required)
            mean_object_features: float32 Tensor, average pooled object localization features, 
                with shape [batch_size, num_objects, feature_depth] (required)
            attention_method: string, either 'softmax' or 'sigmoid' (optional)
            **kwargs: Dict, keyword named properties for common layer attributes, like
                `trainable` etc when constructing the cell from configs of get_config().
            When restoring from CudnnLSTM-trained checkpoints, use
                `CudnnCompatibleLSTMCell` instead.
        """
        
        super(UpDownCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        
        # These must be provided for correct functionality
        if mean_global_features is None:
            raise Exception('mean_global_features must be assigned.')
        if mean_object_features is None:
            raise Exception('mean_object_features must be assigned.')
        if attention_method not in ['softmax', 'sigmoid']:
            raise Exception('attention_method must be in [\'softmax\', \'sigmoid\'].')
        self.mgf = mean_global_features
        self.mof = mean_object_features
        self.batch_size = tf.shape(self.mof)[0] 
        self.num_objects = tf.shape(self.mof)[1] 
        self.feature_size = tf.shape(self.mof)[2]
        self.attn_fn = {'sigmoid': _sigmoid_attention, 
            'softmax': _softmax_attention}[attention_method]
        self._initializer = initializers.get(initializer)  
        
        # Create an LSTM an LSTM that attends to image features.
        self.visual_lstm = tf.contrib.rnn.LSTMCell(
            num_units, use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=True,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        
        # Create an LSTM that predicts the next word token
        self.language_lstm = tf.contrib.rnn.LSTMCell(
            num_units, use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=True,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        
        # Create a spatial attention mechanism.
        self.attn_layer = tf.layers.Conv1D(1, 3, kernel_initializer=self._initializer, 
            padding="same", activation=self.attn_fn, name="attention")
        
        # For managing the RNN functions such as 'zero_state'
        self._state_size = UpDownStateTuple(
            self.visual_lstm.state_size, self.language_lstm.state_size)
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        
        # Compute the bottom-up and top-down attention mechanisms
        v_inputs = tf.concat([tf.concat(state.v, 1), self.mgf, inputs], 1)
        v_outputs, v_next_state = self.visual_lstm(v_inputs, state.v)
        attn_inputs = tf.concat([self.mof, _tile_new_axis(v_outputs, [self.num_objects], [1])], 2)
        attended_mof = tf.reduce_sum(self.mof * self.attn_layer(attn_inputs), 1)
        l_inputs = tf.concat([v_outputs, attended_mof, inputs], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state.l)
        return l_outputs, UpDownStateTuple(v_next_state, l_next_state)
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib import layers
from tensorflow import concat
from tensorflow import variable_scope
from tensorflow import nn
from tensorflow import clip_by_value

class DRNNCell(RNNCell):
  """The most basic RNN cell.
  Args:
    num_units: [int], The list of unit number in layer.
    num_output: int, The number of units for output layer
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_output, num_units, activation=nn.tanh, output_activation=None, reuse=None, phase=True):
    super(DRNNCell, self).__init__(_reuse=reuse)
    self._num_layers = len(num_units)
    self._num_units = num_units
    self._num_output = num_output
    self._activation = activation
    self._output_activation = output_activation
    self.phase = phase

  @property
  def state_size(self):
    return self._num_output

  @property
  def output_size(self):
    return self._num_output

  def call(self, inputs, state):

    s = concat([inputs,state],1)

    hidden = []

    scope_name = "Layer1" 
    fc = layers.fully_connected(s, self._num_units[0], activation_fn=self._activation, scope=scope_name) 
    bn = layers.batch_norm(fc, center=True, scale=True, is_training=self.phase, scope=scope_name+"_bn" )
    hidden.append( bn )

    for l in range(1,self._num_layers):
        scope_name = "Layer{0}".format(l+1) 
        fc = layers.fully_connected(hidden[l-1], self._num_units[l], activation_fn=self._activation, scope=scope_name)
        bn = layers.batch_norm(fc, center=True, scale=True, is_training=self.phase, scope=scope_name+"_bn" )
        hidden.append( bn )

    scope_name = "Layer{0}".format(self._num_layers+1)
    output = layers.fully_connected(hidden[-1], self._num_output, activation_fn=self._output_activation, scope=scope_name)
    output_sat = clip_by_value(output, clip_value_min = 0, 
                                          clip_value_max = 1,
                                          name="output_saturation" )
    return output, output


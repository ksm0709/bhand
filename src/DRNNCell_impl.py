from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib import layers
from tensorflow import concat
from tensorflow import variable_scope
from tensorflow import nn

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

  def __init__(self, num_output, num_units, activation=None, reuse=None, keep_prob=1.0):
    super(DRNNCell, self).__init__(_reuse=reuse)
    self._num_layers = len(num_units)
    self._num_units = num_units
    self._num_output = num_output
    self._activation = activation or math_ops.tanh
    self._keep_prob = keep_prob

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
    hidden.append( layers.fully_connected(s, self._num_units[0], activation_fn=self._activation, scope=scope_name) )

    for l in range(1,self._num_layers):
        scope_name = "Layer{0}".format(l+1) 
        hidden.append( layers.fully_connected(hidden[l-1], self._num_units[l], activation_fn=self._activation, scope=scope_name) )

    scope_name = "Layer{0}".format(self._num_layers+1)
    output = layers.fully_connected(hidden[-1], self._num_output, activation_fn=None, scope=scope_name)
    output_dropout = nn.dropout(output, keep_prob=self._keep_prob)

    return output, output


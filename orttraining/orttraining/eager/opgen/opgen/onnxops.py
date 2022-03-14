# AUTO-GENERATED CODE! - DO NOT EDIT!
# $ python onnxgen.py

from opgen.generator import ONNXAttr, ONNXOp, AttrType

class Abs(ONNXOp):
  """
  Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the absolute is, y = abs(x), is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Abs', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      X)

class Acos(ONNXOp):
  """
  Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Acos', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Acosh(ONNXOp):
  """
  Calculates the hyperbolic arccosine of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Acosh', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Adagrad(ONNXOp):
  """
      Compute one iteration of ADAGRAD, a stochastic gradient based optimization
      algorithm. This operator can conduct the optimization of multiple tensor variables.
  
      Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
      some parameters:
  
       - The initial learning-rate "R".
       - The update count "T". That is, the number of training iterations conducted.
       - A L2-norm regularization coefficient "norm_coefficient".
       - A learning-rate decay factor "decay_factor".
       - A small constant "epsilon" to avoid dividing-by-zero.
  
      At each ADAGRAD iteration, the optimized tensors are moved along a direction
      computed based on their estimated gradient and accumulated squared gradient. Assume
      that only a single tensor "X" is updated by this operator. We need the value of "X",
      its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
      this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
      parameters are given as attributes because they are usually constants. Also, the
      corresponding output tensors are the new value of "X" (called "X_new"), and then
      the new accumulated squared gradient (called "H_new"). Those outputs are computed
      from the given inputs following the pseudo code below.
  
      Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
      numpy-style broadcasting support. The pseudo code to compute those outputs is:
  
        // Compute a scalar learning-rate factor. At the first update of X, T is generally
        // 0 (0-based update index) or 1 (1-based update index).
        r = R / (1 + T * decay_factor);
  
        // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
        G_regularized = norm_coefficient * X + G;
  
        // Compute new accumulated squared gradient.
        H_new = H + G_regularized * G_regularized;
  
        // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
        // computes element-wise square-root.
        H_adaptive = Sqrt(H_new) + epsilon
  
        // Compute the new value of "X".
        X_new = X - r * G_regularized / H_adaptive;
  
      If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
      pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
      concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
      be concatenated too) and then just reuse the entire pseudo code.
  
      Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
      In that reference paper, this operator is a special case of the Figure 1's composite mirror
      descent update.
  """

  def __init__(self, R, T, inputs,
    decay_factor=None, 
    epsilon=None, 
    norm_coefficient=None):
    super().__init__('Adagrad', 1,
      [{'at::kDouble', 'at::kFloat'}, {'at::kLong'}, {'at::kDouble', 'at::kFloat'}],
      R,T,inputs,
      decay_factor=ONNXAttr(decay_factor, AttrType.FLOAT), 
      epsilon=ONNXAttr(epsilon, AttrType.FLOAT), 
      norm_coefficient=ONNXAttr(norm_coefficient, AttrType.FLOAT))

class Adam(ONNXOp):
  """
      Compute one iteration of Adam, a stochastic gradient based optimization
      algorithm. This operator can conduct the optimization of multiple tensor variables.
  
      Let's define the behavior of this operator. First of all, Adam requires
      some parameters:
  
       - The learning-rate "R".
       - The update count "T". That is, the number of training iterations conducted.
       - A L2-norm regularization coefficient "norm_coefficient".
       - A small constant "epsilon" to avoid dividing-by-zero.
       - Two coefficients, "alpha" and "beta".
  
      At each Adam iteration, the optimized tensors are moved along a direction
      computed based on their exponentially-averaged historical gradient and
      exponentially-averaged historical squared gradient. Assume that only a tensor
      "X" is being optimized. The rest of required information is
  
       - the value of "X",
       - "X"'s gradient (denoted by "G"),
       - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
       - "X"'s exponentially-averaged historical squared gradient (denoted by "H").
  
      Some of those parameters are passed into this operator as input tensors and others
      are stored as this operator's attributes. Specifically, this operator's input tensor
      list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
      the second input, and so on. Other parameters are given as attributes because they
      are constants. Moreover, the corresponding output tensors are
  
       - the new value of "X" (called "X_new"),
       - the new exponentially-averaged historical gradient (denoted by "V_new"), and
       - the new exponentially-averaged historical squared gradient (denoted by "H_new").
  
      Those outputs are computed following the pseudo code below.
  
      Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
      numpy-style broadcasting support. The pseudo code to compute those outputs is:
  
        // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
        G_regularized = norm_coefficient * X + G
  
        // Update exponentially-averaged historical gradient.
        V_new = alpha * V + (1 - alpha) * G_regularized
  
        // Update exponentially-averaged historical squared gradient.
        H_new = beta * H + (1 - beta) * G_regularized * G_regularized
  
        // Compute the element-wise square-root of H_new. V_new will be element-wisely
        // divided by H_sqrt for a better update direction.
        H_sqrt = Sqrt(H_new) + epsilon
  
        // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
        R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R
  
        // Compute new value of "X".
        X_new = X - R_adjusted * V_new / H_sqrt
  
        // Post-update regularization.
        X_final = (1 - norm_coefficient_post) * X_new
  
      If there are multiple inputs to be optimized, the pseudo code will be applied
      independently to each of them.
  """

  def __init__(self, R, T, inputs,
    alpha=None, 
    beta=None, 
    epsilon=None, 
    norm_coefficient=None, 
    norm_coefficient_post=None):
    super().__init__('Adam', 1,
      [{'at::kDouble', 'at::kFloat'}, {'at::kLong'}, {'at::kDouble', 'at::kFloat'}],
      R,T,inputs,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      beta=ONNXAttr(beta, AttrType.FLOAT), 
      epsilon=ONNXAttr(epsilon, AttrType.FLOAT), 
      norm_coefficient=ONNXAttr(norm_coefficient, AttrType.FLOAT), 
      norm_coefficient_post=ONNXAttr(norm_coefficient_post, AttrType.FLOAT))

class Add(ONNXOp):
  """
  Performs element-wise binary addition (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  """

  def __init__(self, A, B):
    super().__init__('Add', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class And(ONNXOp):
  """
  Returns the tensor resulted from performing the `and` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('And', 1,
      [{'at::kBool'}, {'at::kBool'}],
      A,B)

class ArgMax(ONNXOp):
  """
  Computes the indices of the max elements of the input tensor's element along the
  provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
  If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
  If select_last_index is True (default False), the index of the last occurrence of the max
  is selected if the max appears more than once in the input. Otherwise the index of the
  first occurrence is selected.
  The type of the output tensor is integer.
  """

  def __init__(self, data,
    axis=None, 
    keepdims=None, 
    select_last_index=None):
    super().__init__('ArgMax', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axis=ONNXAttr(axis, AttrType.INT), 
      keepdims=ONNXAttr(keepdims, AttrType.INT), 
      select_last_index=ONNXAttr(select_last_index, AttrType.INT))

class ArgMin(ONNXOp):
  """
  Computes the indices of the min elements of the input tensor's element along the
  provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
  If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
  If select_last_index is True (default False), the index of the last occurrence of the min
  is selected if the min appears more than once in the input. Otherwise the index of the
  first occurrence is selected.
  The type of the output tensor is integer.
  """

  def __init__(self, data,
    axis=None, 
    keepdims=None, 
    select_last_index=None):
    super().__init__('ArgMin', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axis=ONNXAttr(axis, AttrType.INT), 
      keepdims=ONNXAttr(keepdims, AttrType.INT), 
      select_last_index=ONNXAttr(select_last_index, AttrType.INT))

class ArrayFeatureExtractor(ONNXOp):
  """
      Select elements of the input tensor based on the indices passed.<br>
      The indices are applied to the last axes of the tensor.
  """

  def __init__(self, X, Y):
    super().__init__('ArrayFeatureExtractor', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}, {'at::kLong'}],
      X,Y)

class Asin(ONNXOp):
  """
  Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Asin', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Asinh(ONNXOp):
  """
  Calculates the hyperbolic arcsine of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Asinh', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Atan(ONNXOp):
  """
  Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Atan', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Atanh(ONNXOp):
  """
  Calculates the hyperbolic arctangent of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Atanh', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class AveragePool(ONNXOp):
  """
   AveragePool consumes an input tensor X and applies average pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   average pooling consisting of computing the average on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
   ```
   or
   ```
   output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
   ```
   if ceil_mode is enabled
  
   ```
   * pad_shape[i] is sum of pads along axis i
   ```
  
   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
   ```
   The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
   
  """

  def __init__(self, X,
    auto_pad=None, 
    ceil_mode=None, 
    count_include_pad=None, 
    kernel_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('AveragePool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      ceil_mode=ONNXAttr(ceil_mode, AttrType.INT), 
      count_include_pad=ONNXAttr(count_include_pad, AttrType.INT), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class BatchNormalization(ONNXOp):
  """
  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  There are five required inputs 'X', 'scale', 'B', 'input_mean' and
  'input_var'.
  Note that 'input_mean' and 'input_var' are expected to be the estimated
  statistics in inference mode (training_mode=False, default),
  and the running statistics in training mode (training_mode=True).
  There are multiple cases for the number of outputs, which we list below:
  
  Output case #1: Y, running_mean, running_var (training_mode=True)
  Output case #2: Y (training_mode=False)
  
  When training_mode=False, extra outputs are invalid.
  The outputs are updated as follows when training_mode=True:
  ```
  running_mean = input_mean * momentum + current_mean * (1 - momentum)
  running_var = input_var * momentum + current_var * (1 - momentum)
  
  Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
  
  where:
  
  current_mean = ReduceMean(X, axis=all_except_channel_index)
  current_var =  ReduceVar(X, axis=all_except_channel_index)
  
  Notice that ReduceVar refers to the population variance, and it equals to
  sum(sqrd(x_i - x_avg)) / N
  where N is the population size (this formula does not use sample size N - 1).
  
  ```
  
  When training_mode=False:
  ```
  Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
  ```
  
  For previous (depreciated) non-spatial cases, implementors are suggested
  to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, X, scale, B, input_mean, input_var,
    epsilon=None, 
    momentum=None, 
    training_mode=None):
    super().__init__('BatchNormalization', 3,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X,scale,B,input_mean,input_var,
      epsilon=ONNXAttr(epsilon, AttrType.FLOAT), 
      momentum=ONNXAttr(momentum, AttrType.FLOAT), 
      training_mode=ONNXAttr(training_mode, AttrType.INT))

class Binarizer(ONNXOp):
  """
      Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
  """

  def __init__(self, X,
    threshold=None):
    super().__init__('Binarizer', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      threshold=ONNXAttr(threshold, AttrType.FLOAT))

class BitShift(ONNXOp):
  """
  Bitwise shift operator performs element-wise operation. For each input element, if the
   attribute "direction" is "RIGHT", this operator moves its binary representation toward
   the right side so that the input value is effectively decreased. If the attribute "direction"
   is "LEFT", bits of binary representation moves toward the left side, which results the
   increase of its actual value. The input X is the tensor to be shifted and another input
   Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
   and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
   X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].
  
   Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
   not necessarily identical.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, X, Y,
    direction=None):
    super().__init__('BitShift', 1,
      [set(), set()],
      X,Y,
      direction=ONNXAttr(direction, AttrType.STRING))

class Cast(ONNXOp):
  """
  The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type. The 'to' argument must be one of the data types specified
  in the 'DataType' enum field in the TensorProto message.
  
  Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
  (e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
  result 100. There are some string literals reserved for special floating-point values;
  "+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
  Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
  this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
  to string tensors, plain floating-point representation (such as "314.15926") would be used.
  Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
  of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.
  
  Conversion from a numerical type to any numerical type is always allowed.
  User must be aware of precision loss and value change caused by range difference between two types.
  For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
  an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
  """

  def __init__(self, input,
    to=None):
    super().__init__('Cast', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      input,
      to=ONNXAttr(to, AttrType.INT))

class CastMap(ONNXOp):
  """
      Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
      in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
      If using sparse packing, the key cannot exceed the max_map-1 value.
  """

  def __init__(self, X,
    cast_to=None, 
    map_form=None, 
    max_map=None):
    super().__init__('CastMap', 1,
      [set()],
      X,
      cast_to=ONNXAttr(cast_to, AttrType.STRING), 
      map_form=ONNXAttr(map_form, AttrType.STRING), 
      max_map=ONNXAttr(max_map, AttrType.INT))

class CategoryMapper(ONNXOp):
  """
      Converts strings to integers and vice versa.<br>
      Two sequences of equal length are used to map between integers and strings,
      with strings and integers at the same index detailing the mapping.<br>
      Each operator converts either integers to strings or strings to integers, depending
      on which default value attribute is provided. Only one default value attribute
      should be defined.<br>
      If the string default value is set, it will convert integers to strings.
      If the int default value is set, it will convert strings to integers.
  """

  def __init__(self, X,
    cats_int64s=None, 
    cats_strings=None, 
    default_int64=None, 
    default_string=None):
    super().__init__('CategoryMapper', 1,
      [{'at::kLong'}],
      X,
      cats_int64s=ONNXAttr(cats_int64s, AttrType.INTS), 
      cats_strings=ONNXAttr(cats_strings, AttrType.STRINGS), 
      default_int64=ONNXAttr(default_int64, AttrType.INT), 
      default_string=ONNXAttr(default_string, AttrType.STRING))

class Ceil(ONNXOp):
  """
  Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Ceil', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class Celu(ONNXOp):
  """
  Continuously Differentiable Exponential Linear Units:
  Perform the linear unit element-wise on the input tensor X
  using formula:
  
  ```
  max(0,x) + min(0,alpha*(exp(x/alpha)-1))
  ```
  """

  def __init__(self, X,
    alpha=None):
    super().__init__('Celu', 1,
      [{'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT))

class Clip(ONNXOp):
  """
  Clip operator limits the given input within an interval. The interval is
  specified by the inputs 'min' and 'max'. They default to
  numeric_limits::lowest() and numeric_limits::max(), respectively.
  """

  def __init__(self, input, min, max):
    super().__init__('Clip', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      input,min,max)

class Compress(ONNXOp):
  """
      Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
      In case axis is not provided, input is flattened before elements are selected.
      Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
      
  """

  def __init__(self, input, condition,
    axis=None):
    super().__init__('Compress', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kBool'}],
      input,condition,
      axis=ONNXAttr(axis, AttrType.INT))

class Concat(ONNXOp):
  """
  Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
  """

  def __init__(self, inputs,
    axis=None):
    super().__init__('Concat', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      inputs,
      axis=ONNXAttr(axis, AttrType.INT))

class ConcatFromSequence(ONNXOp):
  """
  Concatenate a sequence of tensors into a single tensor.
  All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
  By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
  When 'new_axis' is 1, the behavior is similar to numpy.stack.
  """

  def __init__(self, input_sequence,
    axis=None, 
    new_axis=None):
    super().__init__('ConcatFromSequence', 1,
      [set()],
      input_sequence,
      axis=ONNXAttr(axis, AttrType.INT), 
      new_axis=ONNXAttr(new_axis, AttrType.INT))

class Constant(ONNXOp):
  """
  This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
  or value_* must be specified.
  """

  def __init__(self,
    sparse_value=None, 
    value=None, 
    value_float=None, 
    value_floats=None, 
    value_int=None, 
    value_ints=None, 
    value_string=None, 
    value_strings=None):
    super().__init__('Constant', 1,
      [],
      sparse_value=ONNXAttr(sparse_value, AttrType.SPARSE_TENSOR), 
      value=ONNXAttr(value, AttrType.TENSOR), 
      value_float=ONNXAttr(value_float, AttrType.FLOAT), 
      value_floats=ONNXAttr(value_floats, AttrType.FLOATS), 
      value_int=ONNXAttr(value_int, AttrType.INT), 
      value_ints=ONNXAttr(value_ints, AttrType.INTS), 
      value_string=ONNXAttr(value_string, AttrType.STRING), 
      value_strings=ONNXAttr(value_strings, AttrType.STRINGS))

class ConstantOfShape(ONNXOp):
  """
  Generate a tensor with given value and shape.
  """

  def __init__(self, input,
    value=None):
    super().__init__('ConstantOfShape', 1,
      [{'at::kLong'}],
      input,
      value=ONNXAttr(value, AttrType.TENSOR))

class Conv(ONNXOp):
  """
  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  """

  def __init__(self, X, W, B,
    auto_pad=None, 
    dilations=None, 
    group=None, 
    kernel_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('Conv', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,W,B,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      dilations=ONNXAttr(dilations, AttrType.INTS), 
      group=ONNXAttr(group, AttrType.INT), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class ConvInteger(ONNXOp):
  """
  The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
  and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  """

  def __init__(self, x, w, x_zero_point, w_zero_point,
    auto_pad=None, 
    dilations=None, 
    group=None, 
    kernel_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('ConvInteger', 1,
      [{'at::kByte'}, {'at::kByte'}, {'at::kByte'}, {'at::kByte'}],
      x,w,x_zero_point,w_zero_point,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      dilations=ONNXAttr(dilations, AttrType.INTS), 
      group=ONNXAttr(group, AttrType.INT), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class ConvTranspose(ONNXOp):
  """
  The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  
  If the pads parameter is provided the shape of the output is calculated via the following equation:
  
    output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
  
  output_shape can also be explicitly specified in which case pads values are auto generated using these equations:
  
    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
  
      
  """

  def __init__(self, X, W, B,
    auto_pad=None, 
    dilations=None, 
    group=None, 
    kernel_shape=None, 
    output_padding=None, 
    output_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('ConvTranspose', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,W,B,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      dilations=ONNXAttr(dilations, AttrType.INTS), 
      group=ONNXAttr(group, AttrType.INT), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      output_padding=ONNXAttr(output_padding, AttrType.INTS), 
      output_shape=ONNXAttr(output_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class Cos(ONNXOp):
  """
  Calculates the cosine of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Cos', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Cosh(ONNXOp):
  """
  Calculates the hyperbolic cosine of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Cosh', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class CumSum(ONNXOp):
  """
  Performs cumulative sum of the input elements along the given axis.
  By default, it will do the sum inclusively meaning the first element is copied as is.
  Through an `exclusive` attribute, this behavior can change to exclude the first element.
  It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.
  
  Example:
  ```
  input_x = [1, 2, 3]
  axis=0
  output = [1, 3, 6]
  exclusive=1
  output = [0, 1, 3]
  exclusive=0
  reverse=1
  output = [6, 5, 3]
  exclusive=1
  reverse=1
  output = [5, 3, 0]
  ```
   
  """

  def __init__(self, x, axis,
    exclusive=None, 
    reverse=None):
    super().__init__('CumSum', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong', 'at::kInt'}],
      x,axis,
      exclusive=ONNXAttr(exclusive, AttrType.INT), 
      reverse=ONNXAttr(reverse, AttrType.INT))

class DepthToSpace(ONNXOp):
  """
  DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
  This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
  the input tensor where values from the depth dimension are moved in spatial blocks to the height
  and width dimensions. By default, `mode` = `DCR`.
  In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
  following order: depth, column, and then row. The output y is computed from the input x as below:
  
  b, c, h, w = x.shape
  
  tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
  
  tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
  
  y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
  
  
  In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
  following order: column, row, and the depth. The output y is computed from the input x as below:
  
  b, c, h, w = x.shape
  
  tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
  
  tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
  
  y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
  """

  def __init__(self, input,
    blocksize=None, 
    mode=None):
    super().__init__('DepthToSpace', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      input,
      blocksize=ONNXAttr(blocksize, AttrType.INT), 
      mode=ONNXAttr(mode, AttrType.STRING))

class DequantizeLinear(ONNXOp):
  """
  The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
  The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
  for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantizations.
  'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
  there's no zero point (zero point is supposed to be 0).
  """

  def __init__(self, x, x_scale, x_zero_point,
    axis=None):
    super().__init__('DequantizeLinear', 1,
      [{'at::kByte', 'at::kInt'}, {'at::kFloat'}, {'at::kByte', 'at::kInt'}],
      x,x_scale,x_zero_point,
      axis=ONNXAttr(axis, AttrType.INT))

class Det(ONNXOp):
  """
  Det calculates determinant of a square matrix or batches of square matrices.
  Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
  and the inner-most 2 dimensions form square matrices.
  The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
  e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
  """

  def __init__(self, X):
    super().__init__('Det', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class DictVectorizer(ONNXOp):
  """
      Uses an index mapping to convert a dictionary to an array.<br>
      Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
      the key type. The index into the vocabulary array at which the key is found is then
      used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
      The key type of the input map must correspond to the element type of the defined vocabulary attribute.
      Therefore, the output array will be equal in length to the index mapping vector parameter.
      All keys in the input dictionary must be present in the index mapping vector.
      For each item in the input dictionary, insert its value in the output array.
      Any keys not present in the input dictionary, will be zero in the output array.<br>
      For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
      then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
      
  """

  def __init__(self, X,
    int64_vocabulary=None, 
    string_vocabulary=None):
    super().__init__('DictVectorizer', 1,
      [set()],
      X,
      int64_vocabulary=ONNXAttr(int64_vocabulary, AttrType.INTS), 
      string_vocabulary=ONNXAttr(string_vocabulary, AttrType.STRINGS))

class Div(ONNXOp):
  """
  Performs element-wise binary division (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  """

  def __init__(self, A, B):
    super().__init__('Div', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class Dropout(ONNXOp):
  """
  Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
  output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
  Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
  the user can simply not pass `training_mode` input or set it to false.
  ```
  output = scale * data * mask,
  ```
  where
  ```
  scale = 1. / (1. - ratio).
  ```
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, data, ratio, training_mode,
    seed=None):
    super().__init__('Dropout', 2,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kBool'}],
      data,ratio,training_mode,
      seed=ONNXAttr(seed, AttrType.INT))

class DynamicQuantizeLinear(ONNXOp):
  """
  A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
  Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
  Scale is calculated as:
  ```
   y_scale = (max(x) - min(x))/(qmax - qmin)
   * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
   * data range is adjusted to include 0.
  ```
  Zero point is calculated as:
  ```
  intermediate_zero_point = qmin - min(x)/y_scale
  y_zero_point = cast(round(saturate(itermediate_zero_point)))
  * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
  * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
  * rounding to nearest ties to even.
  ```
  Data quantization formula is:
  ```
  y = saturate (round (x / y_scale) + y_zero_point)
  * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
  * rounding to nearest ties to even.
  ```
  """

  def __init__(self, x):
    super().__init__('DynamicQuantizeLinear', 3,
      [{'at::kFloat'}],
      x)

class Einsum(ONNXOp):
  """
  An einsum of the form ```term1, term2 -> output-term``` produces an output tensor using the following equation
  
  ```output[output-term] = reduce-sum( input1[term1] * input2[term] )```
  
  where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
  that do not occur in the output-term.
  
  The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
  convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
  an operand tensor, and the characters within the terms correspond to operands dimensions.
  
  This sequence may be followed by "->" to separate the left and right hand side of the equation.
  If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
  summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
  output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
  equation.
  
  When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
  
  The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
  Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
  The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
  beginning of the output. The equation string may contain space (U+0020) character.
  """

  def __init__(self, Inputs,
    equation=None):
    super().__init__('Einsum', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}],
      Inputs,
      equation=ONNXAttr(equation, AttrType.STRING))

class Elu(ONNXOp):
  """
  Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
  0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
  """

  def __init__(self, X,
    alpha=None):
    super().__init__('Elu', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT))

class Equal(ONNXOp):
  """
  Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('Equal', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class Erf(ONNXOp):
  """
  Computes the error function of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Erf', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      input)

class Exp(ONNXOp):
  """
  Calculates the exponential of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Exp', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input)

class Expand(ONNXOp):
  """
  Broadcast the input tensor following the given shape and the broadcast rule.
  The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
  Dimensions are right alignment;
  Two corresponding dimension must have the same value, or one of them is equal to 1.
  Also, this operator is similar to numpy.broadcast_to(input, shape),
  but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
  It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
  or the shape.ndim < input.shape.ndim.
  """

  def __init__(self, input, shape):
    super().__init__('Expand', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      input,shape)

class EyeLike(ONNXOp):
  """
  Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
  tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
  same as the input tensor. The data type can be specified by the 'dtype' argument. If
  'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
  is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
  The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
  TensorProto message and be valid as an output type.
  """

  def __init__(self, input,
    dtype=None, 
    k=None):
    super().__init__('EyeLike', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      input,
      dtype=ONNXAttr(dtype, AttrType.INT), 
      k=ONNXAttr(k, AttrType.INT))

class FeatureVectorizer(ONNXOp):
  """
      Concatenates input tensors into one continuous output.<br>
      All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
      Inputs are copied to the output maintaining the order of the input arguments.<br>
      All inputs must be integers or floats, while the output will be all floating point values.
  """

  def __init__(self, X,
    inputdimensions=None):
    super().__init__('FeatureVectorizer', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      inputdimensions=ONNXAttr(inputdimensions, AttrType.INTS))

class Flatten(ONNXOp):
  """
  Flattens the input tensor into a 2D matrix. If input tensor has shape
  (d_0, d_1, ... d_n) then the output will have shape
  (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
  """

  def __init__(self, input,
    axis=None):
    super().__init__('Flatten', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      input,
      axis=ONNXAttr(axis, AttrType.INT))

class Floor(ONNXOp):
  """
  Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Floor', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class Gather(ONNXOp):
  """
  Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
  entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
  them in an output tensor of rank q + (r - 1).
  
  axis = 0 :
  
  Let
  k = indices[i_{0}, ..., i_{q-1}]
  Then
  output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]
  
  ```
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    indices = [
        [0, 1],
        [1, 2],
    ]
    output = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
        ],
    ]
  ```
  axis = 1 :
  
  Let
  k = indices[i_{0}, ..., i_{q-1}]
  Then
  output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]
  
  ```
    data = [
        [1.0, 1.2, 1.9],
        [2.3, 3.4, 3.9],
        [4.5, 5.7, 5.9],
    ]
    indices = [
        [0, 2],
    ]
    axis = 1,
    output = [
            [[1.0, 1.9]],
            [[2.3, 3.9]],
            [[4.5, 5.9]],
    ]
  ```
  """

  def __init__(self, data, indices,
    axis=None):
    super().__init__('Gather', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong', 'at::kInt'}],
      data,indices,
      axis=ONNXAttr(axis, AttrType.INT))

class GatherElements(ONNXOp):
  """
  GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
  and an optional attribute `axis` that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0). It is an indexing operation
  that produces its output by indexing into the input data tensor at index
  positions determined by elements of the `indices` tensor.
  Its output shape is the same as the shape of `indices` and consists of one value
  (gathered from the `data`) for each element in `indices`.
  
  For instance, in the 3-D case (r = 3), the output produced is determined
  by the following equations:
  ```
    out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
    out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
    out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
  ```
  
  This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.
  
  Example 1:
  ```
    data = [
        [1, 2],
        [3, 4],
    ]
    indices = [
        [0, 0],
        [1, 0],
    ]
    axis = 1
    output = [
        [
          [1, 1],
          [4, 3],
        ],
    ]
  ```
  Example 2:
  ```
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    indices = [
        [1, 2, 0],
        [2, 0, 0],
    ]
    axis = 0
    output = [
        [
          [4, 8, 3],
          [7, 2, 3],
        ],
    ]
  ```
  """

  def __init__(self, data, indices,
    axis=None):
    super().__init__('GatherElements', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong', 'at::kInt'}],
      data,indices,
      axis=ONNXAttr(axis, AttrType.INT))

class GatherND(ONNXOp):
  """
  Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
  slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.
  
  `indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
  where each element defines a slice of `data`
  
  `batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
  `data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.
  
  Some salient points about the inputs' rank and shape:
  
  1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`
  
  2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.
  
  3) b < min(q, r) is to be honored.
  
  4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)
  
  5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
     It is an error if any of the index values are out of bounds.
  
  The output is computed as follows:
  
  The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.
  
  1) If `indices_shape[-1] > r-b` => error condition
  
  2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
     containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
     of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
     is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)
  
  3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
     containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
     to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
     to form the `output` tensor (Examples 2, 3, 4 and 5 below)
  
  This operator is the inverse of `ScatterND`.
  
  `Example 1`
  
    batch_dims = 0
  
    data    = [[0,1],[2,3]]   # data_shape = [2, 2]
  
    indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
  
    output  = [0,3]           # output_shape = [2]
  
  `Example 2`
  
    batch_dims = 0
  
    data    = [[0,1],[2,3]]  # data_shape = [2, 2]
  
    indices = [[1],[0]]      # indices_shape = [2, 1]
  
    output  = [[2,3],[0,1]]  # output_shape = [2, 2]
  
  `Example 3`
  
    batch_dims = 0
  
    data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]
  
    indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
  
    output  = [[2,3],[4,5]]                 # output_shape = [2, 2]
  
  `Example 4`
  
    batch_dims = 0
  
    data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]
  
    indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
  
    output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]
  
  `Example 5`
  
    batch_dims = 1
  
    data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]
  
    indices = [[1],[0]]             # indices_shape = [2, 1]
  
    output  = [[2,3],[4,5]]             # output_shape = [2, 2]
  """

  def __init__(self, data, indices,
    batch_dims=None):
    super().__init__('GatherND', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      data,indices,
      batch_dims=ONNXAttr(batch_dims, AttrType.INT))

class Gemm(ONNXOp):
  """
  General Matrix multiplication:
  https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
  
  A' = transpose(A) if transA else A
  
  B' = transpose(B) if transB else B
  
  Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
  input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
  and output tensor Y has shape (M, N). A will be transposed before doing the
  computation if attribute transA is non-zero, same for B and transB.
  This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, A, B, C,
    alpha=None, 
    beta=None, 
    transA=None, 
    transB=None):
    super().__init__('Gemm', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      A,B,C,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      beta=ONNXAttr(beta, AttrType.FLOAT), 
      transA=ONNXAttr(transA, AttrType.INT), 
      transB=ONNXAttr(transB, AttrType.INT))

class GlobalAveragePool(ONNXOp):
  """
   GlobalAveragePool consumes an input tensor X and applies average pooling across
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.
  """

  def __init__(self, X):
    super().__init__('GlobalAveragePool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class GlobalLpPool(ONNXOp):
  """
   GlobalLpPool consumes an input tensor X and applies lp pool pooling across
   the values in the same channel. This is equivalent to LpPool with kernel size
   equal to the spatial dimension of input tensor.
  """

  def __init__(self, X,
    p=None):
    super().__init__('GlobalLpPool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      p=ONNXAttr(p, AttrType.INT))

class GlobalMaxPool(ONNXOp):
  """
   GlobalMaxPool consumes an input tensor X and applies max pooling across
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.
  """

  def __init__(self, X):
    super().__init__('GlobalMaxPool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class Gradient(ONNXOp):
  """
  Gradient operator computes the partial derivatives of a specific tensor w.r.t.
  some other tensors. This operator is widely used in gradient-based training
  algorithms. To illustrate its use, let's consider a computation graph,
  
  ```
  X -----.
         |
         v
  W --> Conv --> H --> Gemm --> Y
                        ^
                        |
                        Z
  ```
  
  , where W and Z are trainable tensors. Note that operators' attributes are
  omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
  Y with respect to W (Z). The user can compute gradient by inserting Gradient
  operator to form another graph shown below.
  
  ```
  W --> Conv --> H --> Gemm --> Y
  |      ^              ^
  |      |              |
  |      X              Z
  |      |              |
  |      |   .----------'
  |      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
  |      |   |   "xs" followed by "zs")
  |      v   v
  '---> Gradient(xs=["W", "Z"], zs=["X"], y="Y")
         |   |
         |   '-----------------------------------> dY/dW (1st output of Gradient)
         |
         '---------------------------------------> dY/dZ (2nd output of Gradient)
  ```
  
  By definition, the tensor "y" is a function of independent variables in "xs"
  and "zs". Since we only compute the gradient of "y" w.r.t. the differentiable
  variables in "xs", this Gradient only outputs dY/dW and dY/dZ. Note that "H"
  cannot appear in "xs" and "zs". The reason is that "H" can be determined by
  tensors "W" and "X" and therefore "H" is not an independent variable.
  
  All outputs are optional. If needed, for example, user can assign an empty
  string to the 1st output name of that Gradient to skip the generation of dY/dW.
  Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
  and LSTM.
  
  Gradient operator can compute derivative against intermediate tensors. For
  example, the gradient of Y with respect to H can be done via
  
  ```
  W --> Conv --> H --> Gemm --> Y
         ^       |      ^
         |       |      |
         X       |      Z
         .-------'      |
         |   .----------'
         |   | (H/Z is the 1st/2nd input of Gradient as shown in "xs")
         v   v
        Gradient(xs=["H", "Z"], y="Y")
         |   |
         |   '-----------------------------------> dY/dH (1st output of Gradient)
         |
         '---------------------------------------> dY/dZ (2nd output of Gradient)
  ```
  
  It is possible to represent high-order differentiation using Gradient operators.
  For example, given the following linear model:
  
  ```
  W --> Gemm --> Y --> Loss --> O
         ^              ^
         |              |
         X              L
  ```
  
  To compute the 2nd order derivative of O with respect to W (denoted by
  d^2O/dW^2), one can do
  
  ```
  W --> Gemm --> Y --> Loss --> O
  |      ^              ^
  |      |              |
  |      X .------------L
  |      | |            |
  |      | |            v
  +------+-+> Gradient(xs=["X", "W"], zs=["L"], y="O") ---> dO/dX (1st output of Gradient)
  |      | |    |
  |      | |    '---> dO/dW (2nd output of Gradient)
  |      v v
  '---> Gradient(xs=["X", "W"], zs=["L"], y="dO/dW") ---> d(dO/dW)dX (1st output of
         |                                                  Gradient)
         |
         |
         '---> d^2O/dW^2 (2nd output of Gradient)
  ```
  
  The tensors named in attributes "xs", "zs", and "y" define the differentiated
  computation graph, and the inputs to Gradient node define the values at
  which the gradient is computed. We can feed different tensors to the identified
  graph. For example, one can compute the gradient of Y with respect to H at
  a specific value of H, H_1, by providing that value as an input to the Gradient
  node.
  
  ```
  W --> Conv --> H --> Gemm --> Y
         ^              ^
         |              |
         X              Z
  
            Z_1 (2nd input of Gradient)
             |
             v
  H_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dH when H = H_1 and Y = Y_1.
             |
             '------------------------------> dY/dZ (2nd output of Gradient)
  ```
  
  When the inputs of Gradient are the tensors named in "xs" and "zs", the
  computation can be optimized. More specifically, intermediate variables in
  forward pass can be reused if the gradient is computed via reverse-mode
  auto-differentiation.
  """

  def __init__(self, Inputs,
    xs=None, 
    y=None, 
    zs=None):
    super().__init__('Gradient', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      Inputs,
      xs=ONNXAttr(xs, AttrType.STRINGS), 
      y=ONNXAttr(y, AttrType.STRING), 
      zs=ONNXAttr(zs, AttrType.STRINGS))

class Greater(ONNXOp):
  """
  Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('Greater', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class GreaterOrEqual(ONNXOp):
  """
  Returns the tensor resulted from performing the `greater_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('GreaterOrEqual', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}],
      A,B)

class GRU(ONNXOp):
  """
  Computes an one-layer GRU. This operator is usually supported via some custom
  implementation such as CuDNN.
  
  Notations:
  
  `X` - input tensor
  
  `z` - update gate
  
  `r` - reset gate
  
  `h` - hidden gate
  
  `t` - time step (t-1 means previous time step)
  
  `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
  
  `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
  
  `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
  
  `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
  
  `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
  
  `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
  
  `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
  
  `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
  
  `H` - Hidden state
  
  `num_directions` - 2 if direction == bidirectional else 1
  
  Activation functions:
  
    Relu(x)                - max(0, x)
  
    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  
    Sigmoid(x)             - 1/(1 + e^{-x})
  
    (NOTE: Below are optional)
  
    Affine(x)              - alpha*x + beta
  
    LeakyRelu(x)           - x if x >= 0 else alpha * x
  
    ThresholdedRelu(x)     - x if x >= alpha else 0
  
    ScaledTanh(x)          - alpha*Tanh(beta*x)
  
    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  
    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  
    Softsign(x)            - x/(1 + |x|)
  
    Softplus(x)            - log(1 + e^x)
  
  Equations (Default: f=Sigmoid, g=Tanh):
  
    - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  
    - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  
    - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
  
    - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
  
    - Ht = (1 - zt) (.) ht + zt (.) Ht-1
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, X, W, R, B, sequence_lens, initial_h,
    activation_alpha=None, 
    activation_beta=None, 
    activations=None, 
    clip=None, 
    direction=None, 
    hidden_size=None, 
    layout=None, 
    linear_before_reset=None):
    super().__init__('GRU', 2,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kInt'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,W,R,B,sequence_lens,initial_h,
      activation_alpha=ONNXAttr(activation_alpha, AttrType.FLOATS), 
      activation_beta=ONNXAttr(activation_beta, AttrType.FLOATS), 
      activations=ONNXAttr(activations, AttrType.STRINGS), 
      clip=ONNXAttr(clip, AttrType.FLOAT), 
      direction=ONNXAttr(direction, AttrType.STRING), 
      hidden_size=ONNXAttr(hidden_size, AttrType.INT), 
      layout=ONNXAttr(layout, AttrType.INT), 
      linear_before_reset=ONNXAttr(linear_before_reset, AttrType.INT))

class Hardmax(ONNXOp):
  """
  The operator computes the hardmax values for the given input:
  
   Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise
  
  The input does not need to explicitly be a 2D vector. The "axis" attribute
  indicates the dimension along which Hardmax will be performed.
  The output tensor has the same shape
  and contains the Hardmax values of the corresponding input.
  """

  def __init__(self, input,
    axis=None):
    super().__init__('Hardmax', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input,
      axis=ONNXAttr(axis, AttrType.INT))

class HardSigmoid(ONNXOp):
  """
  HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.
  """

  def __init__(self, X,
    alpha=None, 
    beta=None):
    super().__init__('HardSigmoid', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      beta=ONNXAttr(beta, AttrType.FLOAT))

class HardSwish(ONNXOp):
  """
  HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
  the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
  where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('HardSwish', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class Identity(ONNXOp):
  """
  Identity operator
  """

  def __init__(self, input):
    super().__init__('Identity', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      input)

class If(ONNXOp):
  """
  If conditional
  """

  def __init__(self, cond,
    else_branch=None, 
    then_branch=None):
    super().__init__('If', 1,
      [{'at::kBool'}],
      cond,
      else_branch=ONNXAttr(else_branch, AttrType.GRAPH), 
      then_branch=ONNXAttr(then_branch, AttrType.GRAPH))

class Imputer(ONNXOp):
  """
      Replaces inputs that equal one value with another, leaving all other elements alone.<br>
      This operator is typically used to replace missing values in situations where they have a canonical
      representation, such as -1, 0, NaN, or some extreme value.<br>
      One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
      holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
      width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
      which one depends on whether floats or integers are being processed.<br>
      The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
  """

  def __init__(self, X,
    imputed_value_floats=None, 
    imputed_value_int64s=None, 
    replaced_value_float=None, 
    replaced_value_int64=None):
    super().__init__('Imputer', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      imputed_value_floats=ONNXAttr(imputed_value_floats, AttrType.FLOATS), 
      imputed_value_int64s=ONNXAttr(imputed_value_int64s, AttrType.INTS), 
      replaced_value_float=ONNXAttr(replaced_value_float, AttrType.FLOAT), 
      replaced_value_int64=ONNXAttr(replaced_value_int64, AttrType.INT))

class InstanceNormalization(ONNXOp):
  """
  Carries out instance normalization as described in the paper
  https://arxiv.org/abs/1607.08022.
  
  y = scale * (x - mean) / sqrt(variance + epsilon) + B,
  where mean and variance are computed per instance per channel.
  """

  def __init__(self, input, scale, B,
    epsilon=None):
    super().__init__('InstanceNormalization', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input,scale,B,
      epsilon=ONNXAttr(epsilon, AttrType.FLOAT))

class IsInf(ONNXOp):
  """
  Map infinity to true and other values to false.
  """

  def __init__(self, X,
    detect_negative=None, 
    detect_positive=None):
    super().__init__('IsInf', 1,
      [{'at::kDouble', 'at::kFloat'}],
      X,
      detect_negative=ONNXAttr(detect_negative, AttrType.INT), 
      detect_positive=ONNXAttr(detect_positive, AttrType.INT))

class IsNaN(ONNXOp):
  """
  Returns which elements of the input are NaN.
  """

  def __init__(self, X):
    super().__init__('IsNaN', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class LabelEncoder(ONNXOp):
  """
      Maps each element in the input tensor to another value.<br>
      The mapping is determined by the two parallel attributes, 'keys_*' and
      'values_*' attribute. The i-th value in the specified 'keys_*' attribute
      would be mapped to the i-th value in the specified 'values_*' attribute. It
      implies that input's element type and the element type of the specified
      'keys_*' should be identical while the output type is identical to the
      specified 'values_*' attribute. If an input element can not be found in the
      specified 'keys_*' attribute, the 'default_*' that matches the specified
      'values_*' attribute may be used as its output value.<br>
      Let's consider an example which maps a string tensor to an integer tensor.
      Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
      and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
      "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
      Since this operator is an one-to-one mapping, its input and output shapes
      are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
      For key look-up, bit-wise comparison is used so even a float NaN can be
      mapped to a value in 'values_*' attribute.<br>
  """

  def __init__(self, X,
    default_float=None, 
    default_int64=None, 
    default_string=None, 
    keys_floats=None, 
    keys_int64s=None, 
    keys_strings=None, 
    values_floats=None, 
    values_int64s=None, 
    values_strings=None):
    super().__init__('LabelEncoder', 1,
      [{'at::kLong', 'at::kFloat'}],
      X,
      default_float=ONNXAttr(default_float, AttrType.FLOAT), 
      default_int64=ONNXAttr(default_int64, AttrType.INT), 
      default_string=ONNXAttr(default_string, AttrType.STRING), 
      keys_floats=ONNXAttr(keys_floats, AttrType.FLOATS), 
      keys_int64s=ONNXAttr(keys_int64s, AttrType.INTS), 
      keys_strings=ONNXAttr(keys_strings, AttrType.STRINGS), 
      values_floats=ONNXAttr(values_floats, AttrType.FLOATS), 
      values_int64s=ONNXAttr(values_int64s, AttrType.INTS), 
      values_strings=ONNXAttr(values_strings, AttrType.STRINGS))

class LeakyRelu(ONNXOp):
  """
  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  """

  def __init__(self, X,
    alpha=None):
    super().__init__('LeakyRelu', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT))

class Less(ONNXOp):
  """
  Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('Less', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class LessOrEqual(ONNXOp):
  """
  Returns the tensor resulted from performing the `less_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('LessOrEqual', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}],
      A,B)

class LinearClassifier(ONNXOp):
  """
      Linear classifier
  """

  def __init__(self, X,
    classlabels_ints=None, 
    classlabels_strings=None, 
    coefficients=None, 
    intercepts=None, 
    multi_class=None, 
    post_transform=None):
    super().__init__('LinearClassifier', 2,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      classlabels_ints=ONNXAttr(classlabels_ints, AttrType.INTS), 
      classlabels_strings=ONNXAttr(classlabels_strings, AttrType.STRINGS), 
      coefficients=ONNXAttr(coefficients, AttrType.FLOATS), 
      intercepts=ONNXAttr(intercepts, AttrType.FLOATS), 
      multi_class=ONNXAttr(multi_class, AttrType.INT), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING))

class LinearRegressor(ONNXOp):
  """
      Generalized linear regression evaluation.<br>
      If targets is set to 1 (default) then univariate regression is performed.<br>
      If targets is set to M then M sets of coefficients must be passed in as a sequence
      and M results will be output for each input n in N.<br>
      The coefficients array is of length n, and the coefficients for each target are contiguous.
      Intercepts are optional but if provided must match the number of targets.
  """

  def __init__(self, X,
    coefficients=None, 
    intercepts=None, 
    post_transform=None, 
    targets=None):
    super().__init__('LinearRegressor', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      coefficients=ONNXAttr(coefficients, AttrType.FLOATS), 
      intercepts=ONNXAttr(intercepts, AttrType.FLOATS), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING), 
      targets=ONNXAttr(targets, AttrType.INT))

class Log(ONNXOp):
  """
  Calculates the natural log of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Log', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input)

class LogSoftmax(ONNXOp):
  """
  The operator computes the log of softmax values for the given input:
  
   LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
  
  The input does not need to explicitly be a 2D vector. The "axis" attribute
  indicates the dimension along which LogSoftmax will be performed.
  The output tensor has the same shape
  and contains the LogSoftmax values of the corresponding input.
  """

  def __init__(self, input,
    axis=None):
    super().__init__('LogSoftmax', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input,
      axis=ONNXAttr(axis, AttrType.INT))

class Loop(ONNXOp):
  """
  Generic Looping construct. This loop has multiple termination conditions:
  
  1) Trip count. Iteration count specified at runtime. Set by
     specifying the input M. Optional. Set to empty string to omit.
     Note that a static trip count (specified at graph construction time) can be
     specified by passing in a constant node for input M.
  2) Loop termination condition. This is an input to the op that determines
     whether to run the first iteration and also a loop-carried dependency for
     the body graph. The body graph must yield a value for the condition variable,
     whether this input is provided or not.
  
  This table summarizes the operating modes of this operator with equivalent
  C-style code:
  
      Operator inputs defined as (max_trip_count, condition_var).
  
      input ("", ""):
          for (int i=0; ; ++i) {
            cond = ... // Note this value is ignored, but is required in the body
          }
  
      input ("", cond) // Note this is analogous to a while loop
          bool cond = ...;
          for (int i=0; cond; ++i) {
            cond = ...;
          }
  
      input ("", 1) // Note this is analogous to a do-while loop
          bool cond = true
          for (int i=0; cond; ++i) {
            cond = ...;
          }
  
      input (trip_count, "") // Note this is analogous to a for loop
          int trip_count = ...
          for (int i=0; i < trip_count; ++i) {
            cond = ...; // ignored
          }
  
      input (trip_count, cond)
          int trip_count = ...;
          bool cond = ...;
          for (int i=0; i < trip_count && cond; ++i) {
            cond = ...;
          }
  
  
  *Sample usage - cond as well as trip count*
  
      graph predict-net {
        %a = Constant[value = <Scalar Tensor [3]>]()
        %b = Constant[value = <Scalar Tensor [6]>]()
        %keepgoing = Constant[value = <Scalar Tensor [1]>]()
        %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
        %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
        return
      }
  
      graph body-net (
        %i[INT32, scalar]           // iteration number
        %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
        %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
      ) {
        %my_local = Add(%a, %b_in)
        %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
        %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
        %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
        return %keepgoing_out, %b_out, %user_defined_val
      }
  
  *Sample equivalent C code*
  
      {
        /* User-defined code (enclosing scope) */
        int a = 3, b = 6;
        bool keepgoing = true; // Analogous to input cond
        /* End user-defined code */
  
        /* Implicitly-defined code */
        const int max_trip_count = 10; // Analogous to input M
        int user_defined_vals[]; // Imagine this is resizable
        /* End implicitly-defined code */
        /* initialize loop-carried variables and scan-output variables */
        bool keepgoing_out = keepgoing
        int b_out = b
  
        for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
          /* Implicitly-defined code: bind actual parameter values
             to formal parameter variables of loop-body */
          bool keepgoing_in = keepgoing_out;
          bool b_in = b_out;
  
          /* User-defined code (loop body) */
          int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
          b_out = a - b_in;
          keepgoing_out = my_local > b_out;
          user_defined_val = b_in + b_in; // b_in and b_out are different variables
          /* End user-defined code */
  
          /* Implicitly defined-code */
          user_defined_vals[i] = user_defined_val // accumulate scan-output values
        }
        // int t = my_local; // Can't do this. my_local is not accessible here.
  
        // The values below are bound to the output variables of the loop and therefore accessible
        // b_out; user_defined_vals; keepgoing_out;
      }
  
  There are several things of note in this code snippet:
  
  1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
     be referenced in the inputs of the loop.
  2) Any values computed in the loop body that needs to be used in a subsequent
     iteration or after the loop are modelled using a pair of variables in the loop-body,
     consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
     These are referred to as loop-carried dependences. The loop operation node
     supplies the input value of the input variable for the first iteration, and
     returns the output value of the output variable produced by the final
     iteration.
  3) Scan_output variables are used to implicitly concatenate values computed across
     all the iterations. In the above example, the value of user_defined_val computed
     over all iterations are concatenated and returned as the value of user_defined_vals
     after the loop.
  4) Values created in the body cannot be accessed in the enclosing scope,
     except using the mechanism described above.
  
  Note that the semantics of this op support "diagonal" or "wavefront" execution.
  (See Step 3 here for an example:
  https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
  Frontends should emit multi-layer RNNs as a series of While operators (with
  time being the inner looping dimension), with each successive layer consuming
  the scan_outputs from the previous layer, possibly going through several
  point-wise operators (e.g. dropout, residual connections, linear layer).
  
  The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
  """

  def __init__(self, M, cond, v_initial,
    body=None):
    super().__init__('Loop', 1,
      [{'at::kLong'}, {'at::kBool'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      M,cond,v_initial,
      body=ONNXAttr(body, AttrType.GRAPH))

class LpNormalization(ONNXOp):
  """
  Given a matrix, apply Lp-normalization along the provided axis.
  """

  def __init__(self, input,
    axis=None, 
    p=None):
    super().__init__('LpNormalization', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input,
      axis=ONNXAttr(axis, AttrType.INT), 
      p=ONNXAttr(p, AttrType.INT))

class LpPool(ONNXOp):
  """
   LpPool consumes an input tensor X and applies Lp pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   Lp pooling consisting of computing the Lp norm on all values of a subset
   of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing.
  """

  def __init__(self, X,
    auto_pad=None, 
    kernel_shape=None, 
    p=None, 
    pads=None, 
    strides=None):
    super().__init__('LpPool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      p=ONNXAttr(p, AttrType.INT), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class LRN(ONNXOp):
  """
  Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
  It normalizes over local input regions.
  The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
  of shape (N x C x D1 x D2, ..., Dk), its region is
  {X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.
  
  square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
  where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).
  
  Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
  """

  def __init__(self, X,
    alpha=None, 
    beta=None, 
    bias=None, 
    size=None):
    super().__init__('LRN', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      beta=ONNXAttr(beta, AttrType.FLOAT), 
      bias=ONNXAttr(bias, AttrType.FLOAT), 
      size=ONNXAttr(size, AttrType.INT))

class LSTM(ONNXOp):
  """
  Computes an one-layer LSTM. This operator is usually supported via some
  custom implementation such as CuDNN.
  
  Notations:
  
  `X` - input tensor
  
  `i` - input gate
  
  `o` - output gate
  
  `f` - forget gate
  
  `c` - cell gate
  
  `t` - time step (t-1 means previous time step)
  
  `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
  
  `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
  
  `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
  
  `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
  
  `P[iof]`  - P peephole weight vector for input, output, and forget gates
  
  `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
  
  `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
  
  `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
  
  `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
  
  `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
  
  `H` - Hidden state
  
  `num_directions` - 2 if direction == bidirectional else 1
  
  Activation functions:
  
    Relu(x)                - max(0, x)
  
    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  
    Sigmoid(x)             - 1/(1 + e^{-x})
  
    (NOTE: Below are optional)
  
    Affine(x)              - alpha*x + beta
  
    LeakyRelu(x)           - x if x >= 0 else alpha * x
  
    ThresholdedRelu(x)     - x if x >= alpha else 0
  
    ScaledTanh(x)          - alpha*Tanh(beta*x)
  
    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  
    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  
    Softsign(x)            - x/(1 + |x|)
  
    Softplus(x)            - log(1 + e^x)
  
  Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
  
    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  
    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  
    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  
    - Ct = ft (.) Ct-1 + it (.) ct
  
    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  
    - Ht = ot (.) h(Ct)
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, X, W, R, B, sequence_lens, initial_h, initial_c, P,
    activation_alpha=None, 
    activation_beta=None, 
    activations=None, 
    clip=None, 
    direction=None, 
    hidden_size=None, 
    input_forget=None, 
    layout=None):
    super().__init__('LSTM', 3,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kInt'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,W,R,B,sequence_lens,initial_h,initial_c,P,
      activation_alpha=ONNXAttr(activation_alpha, AttrType.FLOATS), 
      activation_beta=ONNXAttr(activation_beta, AttrType.FLOATS), 
      activations=ONNXAttr(activations, AttrType.STRINGS), 
      clip=ONNXAttr(clip, AttrType.FLOAT), 
      direction=ONNXAttr(direction, AttrType.STRING), 
      hidden_size=ONNXAttr(hidden_size, AttrType.INT), 
      input_forget=ONNXAttr(input_forget, AttrType.INT), 
      layout=ONNXAttr(layout, AttrType.INT))

class MatMul(ONNXOp):
  """
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
  """

  def __init__(self, A, B):
    super().__init__('MatMul', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class MatMulInteger(ONNXOp):
  """
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
  The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  """

  def __init__(self, A, B, a_zero_point, b_zero_point):
    super().__init__('MatMulInteger', 1,
      [{'at::kByte'}, {'at::kByte'}, {'at::kByte'}, {'at::kByte'}],
      A,B,a_zero_point,b_zero_point)

class Max(ONNXOp):
  """
  Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, data_0):
    super().__init__('Max', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      data_0)

class MaxPool(ONNXOp):
  """
   MaxPool consumes an input tensor X and applies max pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   max pooling consisting of computing the max on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
   ```
   or
   ```
   output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
   ```
   if ceil_mode is enabled
  
   ```
   * pad_shape[i] is sum of pads along axis i
   ```
  
   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
   ```
   The output of each pooling window is maximum number of elements exclude pad. 
   
  """

  def __init__(self, X,
    auto_pad=None, 
    ceil_mode=None, 
    dilations=None, 
    kernel_shape=None, 
    pads=None, 
    storage_order=None, 
    strides=None):
    super().__init__('MaxPool', 2,
      [{'at::kDouble', 'at::kByte', 'at::kHalf', 'at::kFloat'}],
      X,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      ceil_mode=ONNXAttr(ceil_mode, AttrType.INT), 
      dilations=ONNXAttr(dilations, AttrType.INTS), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      storage_order=ONNXAttr(storage_order, AttrType.INT), 
      strides=ONNXAttr(strides, AttrType.INTS))

class MaxRoiPool(ONNXOp):
  """
   ROI max pool consumes an input tensor X and region of interests (RoIs) to
   apply max pooling across each RoI, to produce output 4-D tensor of shape
   (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  """

  def __init__(self, X, rois,
    pooled_shape=None, 
    spatial_scale=None):
    super().__init__('MaxRoiPool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,rois,
      pooled_shape=ONNXAttr(pooled_shape, AttrType.INTS), 
      spatial_scale=ONNXAttr(spatial_scale, AttrType.FLOAT))

class MaxUnpool(ONNXOp):
  """
  MaxUnpool essentially computes the partial inverse of the MaxPool op.
   The input information to this op is typically the the output information from a MaxPool op. The first
   input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
   from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
   to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
   The third (optional) input is a tensor that specifies the output size of the unpooling operation.
  
  MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
   values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
   the result of an unpooling operation should give back the original input to the unpooling op.
  
  MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
   The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
   known/predictable size.
  
  In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
   which define the exact unpooling op. The attributes typically have the same values as the corrsponding
   pooling op that the unpooling op is trying to invert.
  """

  def __init__(self, X, I, output_shape,
    kernel_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('MaxUnpool', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kLong'}, {'at::kLong'}],
      X,I,output_shape,
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class Mean(ONNXOp):
  """
  Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, data_0):
    super().__init__('Mean', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      data_0)

class MeanVarianceNormalization(ONNXOp):
  """
        A MeanVarianceNormalization Function: Perform mean variance normalization
        on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
  """

  def __init__(self, X,
    axes=None):
    super().__init__('MeanVarianceNormalization', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X,
      axes=ONNXAttr(axes, AttrType.INTS))

class Min(ONNXOp):
  """
  Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, data_0):
    super().__init__('Min', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      data_0)

class Mod(ONNXOp):
  """
    Performs element-wise binary modulus (with Numpy-style broadcasting support).
      The sign of the remainder is the same as that of the Divisor.
  
      Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
      (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
      This attribute is set to 0 by default causing the behavior to be like integer mod.
      Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().
  
      If the input type is floating point, then `fmod` attribute must be set to 1.
  
      In case of dividend being zero, the results will be platform dependent.
  
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B,
    fmod=None):
    super().__init__('Mod', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B,
      fmod=ONNXAttr(fmod, AttrType.INT))

class Momentum(ONNXOp):
  """
      Compute one iteration of stochastic gradient update with momentum.
      This operator can conduct the optimization of multiple tensor variables.
  
      Let's define the behavior of this operator. As you can imagine, SG with momentum requires
      several parameters:
  
       - The learning-rate "R".
       - The update count "T". That is, the number of conducted training iterations. It should
         be zero in the first training iteration.
       - A L2-norm regularization coefficient "norm_coefficient".
       - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
       - The scaling coefficient of current gradient "beta".
       - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
         be used.
  
      For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
      Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
      Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
      momentum (called "V_new").
  
      This operator supports two different momentum algorithms. Set the attribute "mode" to
      "nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
      "standard" to use standard momentum. Computation details are described subsequently.
  
      Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.
  
      Pseudo code for SG with standard momentum:
  
        // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
        // values of all elements in X.
        G_regularized = norm_coefficient * X + G
  
        // In the first training iteration, beta should always be 1.
        beta_adjusted = T > 0 ? beta : 1
  
        // Compute the current momentum based on previous momentum and the current gradient.
        V_new = alpha * V + beta_adjusted * G_regularized
  
        // Update X.
        X_new = X - R * V_new
  
      Pseudo code for SG with Nesterov's momentum:
  
        // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
        // values of all elements in X.
        G_regularized = norm_coefficient * X + G;
  
        // In the first training iteration, beta should always be 1.
        beta_adjusted = T > 0 ? beta : 1
  
        // Compute the current momentum based on previous momentum and the current gradient.
        V_new = alpha * V + beta_adjusted * G_regularized;
  
        // Compute final update direction and then update X.
        X_new = X - R * (G_regularized + alpha * V_new)
  
      If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
      pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
      concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
      be concatenated too) and then our pseudo code becomes applicable.
  """

  def __init__(self, R, T, inputs,
    alpha=None, 
    beta=None, 
    mode=None, 
    norm_coefficient=None):
    super().__init__('Momentum', 1,
      [{'at::kDouble', 'at::kFloat'}, {'at::kLong'}, {'at::kDouble', 'at::kFloat'}],
      R,T,inputs,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      beta=ONNXAttr(beta, AttrType.FLOAT), 
      mode=ONNXAttr(mode, AttrType.STRING), 
      norm_coefficient=ONNXAttr(norm_coefficient, AttrType.FLOAT))

class Mul(ONNXOp):
  """
  Performs element-wise binary multiplication (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  """

  def __init__(self, A, B):
    super().__init__('Mul', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class Multinomial(ONNXOp):
  """
  Generate a tensor of samples from a multinomial distribution according to the probabilities
  of each of the possible outcomes.
  """

  def __init__(self, input,
    dtype=None, 
    sample_size=None, 
    seed=None):
    super().__init__('Multinomial', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input,
      dtype=ONNXAttr(dtype, AttrType.INT), 
      sample_size=ONNXAttr(sample_size, AttrType.INT), 
      seed=ONNXAttr(seed, AttrType.FLOAT))

class Neg(ONNXOp):
  """
  Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Neg', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      X)

class NegativeLogLikelihoodLoss(ONNXOp):
  """
  A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
  Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
  The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
  The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
  or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
  The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
  
      loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
  
  When an optional "weight" is provided, the sample loss is calculated as:
  
      loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
  
  loss is zero for the case when target-value equals ignore_index.
  
      loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
  
  If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
  If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
  
      mean(loss), if "weight" is not provided,
  
  or if weight is provided,
  
      sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
  
  If "reduction" attribute is set to "sum", the output is a scalar:
      sum(loss).
  
  See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
  
  Example 1:
  
      // negative log likelihood loss, "none" reduction
      N, C, d1 = 2, 3, 2
      input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
               [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
      target = [[2, 1], [0, 2]]
  
      loss = np.zeros((N, d1))
      for n in range(N):
          for d_1 in range(d1):
              c = target[n][d_1]
              loss[n][d_1] = -input[n][c][d_1]
  
      // print(loss)
      // [[-3. -2.]
      //  [-0. -2.]]
  
  Example 2:
  
      // weighted negative log likelihood loss, sum reduction
      N, C, d1 = 2, 3, 2
      input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
              [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
      target = [[2, 1], [0, 2]]
      weight = [0.2, 0.3, 0.1]
      loss = np.zeros((N, d1))
      for n in range(N):
          for d_1 in range(d1):
              c = target[n][d_1]
              loss[n][d_1] = -input[n][c][d_1] * weight[c]
  
      loss = np.sum(loss)
      // print(loss)
      // -1.1
  
  Example 3:
  
      // weighted negative log likelihood loss, mean reduction
      N, C, d1 = 2, 3, 2
      input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
              [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
      target = [[2, 1], [0, 2]]
      weight = [0.2, 0.3, 0.1]
      loss = np.zeros((N, d1))
      weight_total = 0
      for n in range(N):
          for d_1 in range(d1):
              c = target[n][d_1]
              loss[n][d_1] = -input[n][c][d_1] * weight[c]
              weight_total = weight_total + weight[c]
  
      loss = np.sum(loss) / weight_total
      // print(loss)
      // -1.57
  """

  def __init__(self, input, target, weight,
    ignore_index=None, 
    reduction=None):
    super().__init__('NegativeLogLikelihoodLoss', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kLong', 'at::kInt'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input,target,weight,
      ignore_index=ONNXAttr(ignore_index, AttrType.INT), 
      reduction=ONNXAttr(reduction, AttrType.STRING))

class NonMaxSuppression(ONNXOp):
  """
  Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
  Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
  Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
  orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
  result in the same boxes being selected by the algorithm.
  The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
  The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
  """

  def __init__(self, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
    center_point_box=None):
    super().__init__('NonMaxSuppression', 1,
      [{'at::kFloat'}, {'at::kFloat'}, {'at::kLong'}, {'at::kFloat'}, {'at::kFloat'}],
      boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold,
      center_point_box=ONNXAttr(center_point_box, AttrType.INT))

class NonZero(ONNXOp):
  """
      Returns the indices of the elements that are non-zero
      (in row-major order - by dimension).
      NonZero behaves similar to numpy.nonzero:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
  """

  def __init__(self, X):
    super().__init__('NonZero', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      X)

class Normalizer(ONNXOp):
  """
      Normalize the input.  There are three normalization modes, which have the corresponding formulas,
      defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
  <br>
      Max: Y = X / max(X)<br>
      L1:  Y = X / sum(X)<br>
      L2:  Y = sqrt(X^2 / sum(X^2)}<br>
      In all modes, if the divisor is zero, Y == X.
  <br>
      For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
      of the batch is normalized independently.
  """

  def __init__(self, X,
    norm=None):
    super().__init__('Normalizer', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      norm=ONNXAttr(norm, AttrType.STRING))

class Not(ONNXOp):
  """
  Returns the negation of the input tensor element-wise.
  """

  def __init__(self, X):
    super().__init__('Not', 1,
      [{'at::kBool'}],
      X)

class OneHot(ONNXOp):
  """
      Produces a one-hot tensor based on inputs.
      The locations represented by the index values in the 'indices' input tensor will have 'on_value'
      and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
      are specified as part of required input argument 'values', which is a two-element tensor of format
      [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
      input tensor. The additional dimension is for one-hot representation. The additional dimension will
      be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
      dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
      dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
      as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
      the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
      output tensor.
  
      when axis = 0:
      output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.
  
      when axis = -1:
      output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
  """

  def __init__(self, indices, depth, values,
    axis=None):
    super().__init__('OneHot', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      indices,depth,values,
      axis=ONNXAttr(axis, AttrType.INT))

class OneHotEncoder(ONNXOp):
  """
      Replace each input element with an array of ones and zeros, where a single
      one is placed at the index of the category that was passed in. The total category count
      will determine the size of the extra dimension of the output array Y.<br>
      For example, if we pass a tensor with a single value of 4, and a category count of 8,
      the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
      This operator assumes every input feature is from the same set of categories.<br>
      If the input is a tensor of float, int32, or double, the data will be cast
      to integers and the cats_int64s category list will be used for the lookups.
  """

  def __init__(self, X,
    cats_int64s=None, 
    cats_strings=None, 
    zeros=None):
    super().__init__('OneHotEncoder', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      cats_int64s=ONNXAttr(cats_int64s, AttrType.INTS), 
      cats_strings=ONNXAttr(cats_strings, AttrType.STRINGS), 
      zeros=ONNXAttr(zeros, AttrType.INT))

class Or(ONNXOp):
  """
  Returns the tensor resulted from performing the `or` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('Or', 1,
      [{'at::kBool'}, {'at::kBool'}],
      A,B)

class Pad(ONNXOp):
  """
  Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  
  The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):
  
  1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)
  
  2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
  
  3) `edge` - pads with the edge values of array
  
  
  Example 1 (`constant` mode):
    Insert 0 pads to the beginning of the second dimension.
  
    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
  
    pads = [0, 2, 0, 0]
  
    mode = 'constant'
  
    constant_value = 0.0
  
    output =
    [
        [0.0, 0.0, 1.0, 1.2],
        [0.0, 0.0, 2.3, 3.4],
        [0.0, 0.0, 4.5, 5.7],
    ]
  
  
  Example 2 (`reflect` mode):
    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
  
    pads = [0, 2, 0, 0]
  
    mode = 'reflect'
  
    output =
    [
        [1.0, 1.2, 1.0, 1.2],
        [2.3, 3.4, 2.3, 3.4],
        [4.5, 5.7, 4.5, 5.7],
    ]
  
  
  Example 3 (`edge` mode):
    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
  
    pads = [0, 2, 0, 0]
  
    mode = 'edge'
  
    output =
    [
        [1.0, 1.0, 1.0, 1.2],
        [2.3, 2.3, 2.3, 3.4],
        [4.5, 4.5, 4.5, 5.7],
    ]
  """

  def __init__(self, data, pads, constant_value,
    mode=None):
    super().__init__('Pad', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data,pads,constant_value,
      mode=ONNXAttr(mode, AttrType.STRING))

class Pow(ONNXOp):
  """
  Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, X, Y):
    super().__init__('Pow', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}],
      X,Y)

class PRelu(ONNXOp):
  """
  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, X, slope):
    super().__init__('PRelu', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat'}],
      X,slope)

class QLinearConv(ONNXOp):
  """
  The convolution operator consumes a quantized input tensor, its scale and zero point,
  a quantized filter, its scale and zero point, and output's scale and zero point,
  and computes the quantized output. Each scale and zero-point pair must have same shape.
  It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
  Each input or output and its related zero point must have same type.
  When bias is present it must be quantized using scale = input scale * weight scale and
  zero point as 0.
  """

  def __init__(self, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B,
    auto_pad=None, 
    dilations=None, 
    group=None, 
    kernel_shape=None, 
    pads=None, 
    strides=None):
    super().__init__('QLinearConv', 1,
      [{'at::kByte'}, {'at::kFloat'}, {'at::kByte'}, {'at::kByte'}, {'at::kFloat'}, {'at::kByte'}, {'at::kFloat'}, {'at::kByte'}, {'at::kInt'}],
      x,x_scale,x_zero_point,w,w_scale,w_zero_point,y_scale,y_zero_point,B,
      auto_pad=ONNXAttr(auto_pad, AttrType.STRING), 
      dilations=ONNXAttr(dilations, AttrType.INTS), 
      group=ONNXAttr(group, AttrType.INT), 
      kernel_shape=ONNXAttr(kernel_shape, AttrType.INTS), 
      pads=ONNXAttr(pads, AttrType.INTS), 
      strides=ONNXAttr(strides, AttrType.INTS))

class QLinearMatMul(ONNXOp):
  """
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
  It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output.
  The quantization formula is y = saturate((x / y_scale) + y_zero_point). For (x / y_scale), it is rounding to nearest ties to even.
  Refer to https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point must have same shape.
  They must be either scalar (per tensor) or 1-D tensor (per row for 'a' and per column for 'b'). If scale and zero point are 1-D tensor,
  the number of elements of scale and zero point tensor of input 'a' and output 'y' should be equal to the number of rows of input 'a',
  and the number of elements of scale and zero point tensor of input 'b' should be equal to the number of columns of input 'b'.
  Production must never overflow, and accumulation may overflow if and only if in 32 bits.
  """

  def __init__(self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
    super().__init__('QLinearMatMul', 1,
      [{'at::kByte'}, {'at::kFloat'}, {'at::kByte'}, {'at::kByte'}, {'at::kFloat'}, {'at::kByte'}, {'at::kFloat'}, {'at::kByte'}],
      a,a_scale,a_zero_point,b,b_scale,b_zero_point,y_scale,y_zero_point)

class QuantizeLinear(ONNXOp):
  """
  The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor. The scale factor can be a scalar
  (per-tensor/layer quantization), or a 1-D tensor for per-axis quantization. The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
  For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
  For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
  """

  def __init__(self, x, y_scale, y_zero_point,
    axis=None):
    super().__init__('QuantizeLinear', 1,
      [{'at::kInt', 'at::kFloat'}, {'at::kFloat'}, {'at::kByte'}],
      x,y_scale,y_zero_point,
      axis=ONNXAttr(axis, AttrType.INT))

class RandomNormal(ONNXOp):
  """
  Generate a tensor with random values drawn from a normal distribution. The shape
  of the tensor is specified by the `shape` argument and the parameter of the normal distribution
  specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  """

  def __init__(self,
    dtype=None, 
    mean=None, 
    scale=None, 
    seed=None, 
    shape=None):
    super().__init__('RandomNormal', 1,
      [],
      dtype=ONNXAttr(dtype, AttrType.INT), 
      mean=ONNXAttr(mean, AttrType.FLOAT), 
      scale=ONNXAttr(scale, AttrType.FLOAT), 
      seed=ONNXAttr(seed, AttrType.FLOAT), 
      shape=ONNXAttr(shape, AttrType.INTS))

class RandomNormalLike(ONNXOp):
  """
  Generate a tensor with random values drawn from a normal distribution.
  The shape of the output tensor is copied from the shape of the input tensor,
  and the parameters of the normal distribution are specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
  The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
  TensorProto message, and be valid as an output type.
  """

  def __init__(self, input,
    dtype=None, 
    mean=None, 
    scale=None, 
    seed=None):
    super().__init__('RandomNormalLike', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      input,
      dtype=ONNXAttr(dtype, AttrType.INT), 
      mean=ONNXAttr(mean, AttrType.FLOAT), 
      scale=ONNXAttr(scale, AttrType.FLOAT), 
      seed=ONNXAttr(seed, AttrType.FLOAT))

class RandomUniform(ONNXOp):
  """
  Generate a tensor with random values drawn from a uniform distribution. The shape
  of the tensor is specified by the `shape` argument and the range by `low` and `high`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  TensorProto message.
  """

  def __init__(self,
    dtype=None, 
    high=None, 
    low=None, 
    seed=None, 
    shape=None):
    super().__init__('RandomUniform', 1,
      [],
      dtype=ONNXAttr(dtype, AttrType.INT), 
      high=ONNXAttr(high, AttrType.FLOAT), 
      low=ONNXAttr(low, AttrType.FLOAT), 
      seed=ONNXAttr(seed, AttrType.FLOAT), 
      shape=ONNXAttr(shape, AttrType.INTS))

class RandomUniformLike(ONNXOp):
  """
  Generate a tensor with random values drawn from a uniform distribution.
  The shape of the output tensor is copied from the shape of the input tensor,
  and the parameters of the uniform distribution are specified by `low` and `high`.
  
  The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
  The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
  TensorProto message and be valid as an output type.
  """

  def __init__(self, input,
    dtype=None, 
    high=None, 
    low=None, 
    seed=None):
    super().__init__('RandomUniformLike', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      input,
      dtype=ONNXAttr(dtype, AttrType.INT), 
      high=ONNXAttr(high, AttrType.FLOAT), 
      low=ONNXAttr(low, AttrType.FLOAT), 
      seed=ONNXAttr(seed, AttrType.FLOAT))

class Range(ONNXOp):
  """
  Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
  up to `limit` (exclusive).
  
  The number of elements in the output of range is computed as below-
  
  `number_of_elements = max( ceil( (limit - start) / delta ) , 0 )`
  
  The pseudocode determining the contents of the output is shown below-
  
  `for(int i=0; i<number_of_elements; ++i)`
  
  `{`
  
  `    output[i] =  start + (i * delta);  `
  
  `}`
  
  `Example 1`
  Inputs: start = 3, limit = 9, delta = 3
  Output: [3, 6]
  
  `Example 2`
  Inputs: start = 10, limit = 4, delta = -2
  Output: [10, 8, 6]
  """

  def __init__(self, start, limit, delta):
    super().__init__('Range', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kShort', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kInt', 'at::kShort', 'at::kFloat'}],
      start,limit,delta)

class Reciprocal(ONNXOp):
  """
  Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Reciprocal', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class ReduceL1(ONNXOp):
  """
  Computes the L1 norm of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceL1', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceL2(ONNXOp):
  """
  Computes the L2 norm of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceL2', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceLogSum(ONNXOp):
  """
  Computes the log sum of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceLogSum', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceLogSumExp(ONNXOp):
  """
  Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceLogSumExp', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceMax(ONNXOp):
  """
  Computes the max of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceMax', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceMean(ONNXOp):
  """
  Computes the mean of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceMean', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceMin(ONNXOp):
  """
  Computes the min of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceMin', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceProd(ONNXOp):
  """
  Computes the product of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceProd', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class ReduceSum(ONNXOp):
  """
  Computes the sum of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data, axes,
    keepdims=None, 
    noop_with_empty_axes=None):
    super().__init__('ReduceSum', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      data,axes,
      keepdims=ONNXAttr(keepdims, AttrType.INT), 
      noop_with_empty_axes=ONNXAttr(noop_with_empty_axes, AttrType.INT))

class ReduceSumSquare(ONNXOp):
  """
  Computes the sum square of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.
  
  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.
  """

  def __init__(self, data,
    axes=None, 
    keepdims=None):
    super().__init__('ReduceSumSquare', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kHalf', 'at::kFloat', 'at::kBFloat16'}],
      data,
      axes=ONNXAttr(axes, AttrType.INTS), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class Relu(ONNXOp):
  """
  Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Relu', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      X)

class Reshape(ONNXOp):
  """
  Reshape the input tensor similar to numpy.reshape.
  First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
  At most one dimension of the new shape can be -1. In this case, the value is
  inferred from the size of the tensor and the remaining dimensions. A dimension
  could also be 0, in which case the actual dimension value is unchanged (i.e. taken
  from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
  dimension will be set explicitly to zero (i.e. not taken from input tensor)
  """

  def __init__(self, data, shape,
    allowzero=None):
    super().__init__('Reshape', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      data,shape,
      allowzero=ONNXAttr(allowzero, AttrType.INT))

class Resize(ONNXOp):
  """
  Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
  """

  def __init__(self, X, roi, scales, sizes,
    coordinate_transformation_mode=None, 
    cubic_coeff_a=None, 
    exclude_outside=None, 
    extrapolation_value=None, 
    mode=None, 
    nearest_mode=None):
    super().__init__('Resize', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kFloat'}, {'at::kLong'}],
      X,roi,scales,sizes,
      coordinate_transformation_mode=ONNXAttr(coordinate_transformation_mode, AttrType.STRING), 
      cubic_coeff_a=ONNXAttr(cubic_coeff_a, AttrType.FLOAT), 
      exclude_outside=ONNXAttr(exclude_outside, AttrType.INT), 
      extrapolation_value=ONNXAttr(extrapolation_value, AttrType.FLOAT), 
      mode=ONNXAttr(mode, AttrType.STRING), 
      nearest_mode=ONNXAttr(nearest_mode, AttrType.STRING))

class ReverseSequence(ONNXOp):
  """
  Reverse batch of sequences having different lengths specified by `sequence_lens`.
  
  For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
  and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
  sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.
  
  Example 1:
    input = [[0.0, 4.0, 8.0,  12.0],
             [1.0, 5.0, 9.0,  13.0],
             [2.0, 6.0, 10.0, 14.0],
             [3.0, 7.0, 11.0, 15.0]]
    sequence_lens = [4, 3, 2, 1]
    time_axis = 0
    batch_axis = 1
  
    output = [[3.0, 6.0, 9.0,  12.0],
              [2.0, 5.0, 8.0,  13.0],
              [1.0, 4.0, 10.0, 14.0],
              [0.0, 7.0, 11.0, 15.0]]
  
  Example 2:
    input = [[0.0,  1.0,  2.0,  3.0 ],
             [4.0,  5.0,  6.0,  7.0 ],
             [8.0,  9.0,  10.0, 11.0],
             [12.0, 13.0, 14.0, 15.0]]
    sequence_lens = [1, 2, 3, 4]
    time_axis = 1
    batch_axis = 0
  
    output = [[0.0,  1.0,  2.0,  3.0 ],
              [5.0,  4.0,  6.0,  7.0 ],
              [10.0, 9.0,  8.0,  11.0],
              [15.0, 14.0, 13.0, 12.0]]
  """

  def __init__(self, input, sequence_lens,
    batch_axis=None, 
    time_axis=None):
    super().__init__('ReverseSequence', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kLong'}],
      input,sequence_lens,
      batch_axis=ONNXAttr(batch_axis, AttrType.INT), 
      time_axis=ONNXAttr(time_axis, AttrType.INT))

class RNN(ONNXOp):
  """
  Computes an one-layer simple RNN. This operator is usually supported
  via some custom implementation such as CuDNN.
  
  Notations:
  
  `X` - input tensor
  
  `i` - input gate
  
  `t` - time step (t-1 means previous time step)
  
  `Wi` - W parameter weight matrix for input gate
  
  `Ri` - R recurrence weight matrix for input gate
  
  `Wbi` - W parameter bias vector for input gate
  
  `Rbi` - R parameter bias vector for input gate
  
  `WBi` - W parameter weight matrix for backward input gate
  
  `RBi` - R recurrence weight matrix for backward input gate
  
  `WBbi` - WR bias vectors for backward input gate
  
  `RBbi` - RR bias vectors for backward input gate
  
  `H` - Hidden state
  
  `num_directions` - 2 if direction == bidirectional else 1
  
  Activation functions:
  
    Relu(x)                - max(0, x)
  
    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  
    Sigmoid(x)             - 1/(1 + e^{-x})
  
    (NOTE: Below are optional)
  
    Affine(x)              - alpha*x + beta
  
    LeakyRelu(x)           - x if x >= 0 else alpha * x
  
    ThresholdedRelu(x)     - x if x >= alpha else 0
  
    ScaledTanh(x)          - alpha*Tanh(beta*x)
  
    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  
    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  
    Softsign(x)            - x/(1 + |x|)
  
    Softplus(x)            - log(1 + e^x)
  
  Equations (Default: f=Tanh):
  
    - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
  """

  def __init__(self, X, W, R, B, sequence_lens, initial_h,
    activation_alpha=None, 
    activation_beta=None, 
    activations=None, 
    clip=None, 
    direction=None, 
    hidden_size=None, 
    layout=None):
    super().__init__('RNN', 2,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kInt'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,W,R,B,sequence_lens,initial_h,
      activation_alpha=ONNXAttr(activation_alpha, AttrType.FLOATS), 
      activation_beta=ONNXAttr(activation_beta, AttrType.FLOATS), 
      activations=ONNXAttr(activations, AttrType.STRINGS), 
      clip=ONNXAttr(clip, AttrType.FLOAT), 
      direction=ONNXAttr(direction, AttrType.STRING), 
      hidden_size=ONNXAttr(hidden_size, AttrType.INT), 
      layout=ONNXAttr(layout, AttrType.INT))

class RoiAlign(ONNXOp):
  """
  Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
  RoiAlign consumes an input tensor X and region of interests (rois)
  to apply pooling across each RoI; it produces a 4-D tensor of shape
  (num_rois, C, output_height, output_width).
  
  RoiAlign is proposed to avoid the misalignment by removing
  quantizations while converting from original image into feature
  map and from feature map into RoI feature; in each ROI bin,
  the value of the sampled locations are computed directly
  through bilinear interpolation.
  """

  def __init__(self, X, rois, batch_indices,
    mode=None, 
    output_height=None, 
    output_width=None, 
    sampling_ratio=None, 
    spatial_scale=None):
    super().__init__('RoiAlign', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kDouble', 'at::kHalf', 'at::kFloat'}, {'at::kLong'}],
      X,rois,batch_indices,
      mode=ONNXAttr(mode, AttrType.STRING), 
      output_height=ONNXAttr(output_height, AttrType.INT), 
      output_width=ONNXAttr(output_width, AttrType.INT), 
      sampling_ratio=ONNXAttr(sampling_ratio, AttrType.INT), 
      spatial_scale=ONNXAttr(spatial_scale, AttrType.FLOAT))

class Round(ONNXOp):
  """
  Round takes one input Tensor and rounds the values, element-wise, meaning
  it finds the nearest integer for each value.
  In case of halfs, the rule is to round them to the nearest even integer.
  The output tensor has the same shape and type as the input.
  
  Examples:
  ```
  round([0.9]) = [1.0]
  round([2.5]) = [2.0]
  round([2.3]) = [2.0]
  round([1.5]) = [2.0]
  round([-4.5]) = [-4.0]
  ```
  """

  def __init__(self, X):
    super().__init__('Round', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class Scaler(ONNXOp):
  """
      Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
  """

  def __init__(self, X,
    offset=None, 
    scale=None):
    super().__init__('Scaler', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      offset=ONNXAttr(offset, AttrType.FLOATS), 
      scale=ONNXAttr(scale, AttrType.FLOATS))

class Scan(ONNXOp):
  """
  Scan can be used to iterate over one or more scan_input tensors,
  constructing zero or more scan_output tensors. It combines ideas from general recurrences,
  functional programming constructs such as scan, fold, map, and zip and is intended to enable
  generalizations of RNN-like constructs for sequence-to-sequence processing.
  Other tensors (referred to as state_variables here) can be used to carry a state
  when iterating from one element to another (similar to hidden-state in RNNs, also referred
  to as loop-carried dependences in the context of loops).
  Many common usages involve a single scan_input tensor (where functionality
  similar to scan, fold and map can be obtained). When more than one scan_input is used,
  a behavior similar to zip is obtained.
  
  The attribute body must be a graph, specifying the computation to be performed in
  every iteration. It takes as input the current values of the state_variables and
  the current iterated element of the scan_inputs. It must return the (updated) values
  of the state_variables and zero or more scan_output_element tensors. The values of the
  scan_output_element tensors are concatenated over all the iterations to produce the
  scan_output values of the scan construct (similar to the concatenated intermediate
  hidden-state values of RNN-like constructs). All the output tensors (state_variables as
  well as scan_output_element tensors) are required to have the same shape in each iteration
  of the loop (a restriction imposed to enable efficient memory allocation).
  
  Note that the iterated element passed to the body subgraph does not have a sequence
  axis. It will have a rank one less than the rank of the corresponding scan_input.
  
  The scan operation returns the final values of the state_variables as well as the
  scan_outputs.
  
  The optional attribute scan_input_directions specifies the direction (forward or backward)
  for each scan input. If this attribute is omitted, all sequences are scanned in the forward
  direction. A bidirectional scan may be performed by specifying the same tensor input twice
  in the scan_inputs, once with a forward direction, and once with a backward direction.
  
  The scan_output of the operation is produced by concatenating the scan_output_element
  values produced by the body in each iteration.  The optional attribute scan_output_directions
  specifies the direction in which scan_output is constructed (by appending or prepending the
  scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
  is omitted, the scan_output_element is appended to the scan_output in each iteration.
  
  The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
  If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
  batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
  Note that scanning a non-zero axis may be less efficient than scanning axis zero.
  
  The optional attribute scan_output_axes specifies the axis along which the scan_outputs
  are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
  scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
  value of 1.
  
  Note that because of the ONNX restriction that only the last parameter of an operator can
  be variadic, the initial-states and scan-inputs are listed together as one input parameter.
  Similarly, the final-states and scan-outputs are listed together as one output parameter.
  The attribute num_scan_inputs indicates the number M of scan-inputs.
  
  The behavior of
  
      Scan <
          num_scan_inputs = m,
          body = loop-body,
          scan_input_axes = [axis_1, ..., axis_m]
      > (init_1, ..., init_n, scan_1, ..., scan_m)
  
  is equivalent to the following pseudo-code:
  
      // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
      // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
      sequence_length = scan_1.shape[axis_1];
  
      // initialize state-variables
      st_1 = init_1; ... st_n = init_n;
      // initialize scan-output variables: [] denotes an empty tensor
      scan_out_1 = []; ...; scan_out_k = [];
      // identify number of iterations:
  
      // execute loop
      for (int t = 0; t < sequence_length; ++t) {
          // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
          // of rank one less than T obtained by indexing T at position t along axis k.
          si_1 = scan_1<axis=axis_1>[t];
          ... ;
          si_m = scan_m<axis=axis_m>[t];
          // execute loop-body
          st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
          // accumulate the scan-output elements
          scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
      }
  
      return st_1, ..., st_n, scan_out_1, ..., scan_out_k;
  
  *Sample usage: Encoding RNN using a Scan*
  
  The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
  recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
  be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
  %Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
  values are computed in the outer graph, they need to be passed in as extra state_variables.
  
      graph rnn-encoding {
        %H_0 = ...
        %X = ...
        %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
        return %Y, %Y_h
      }
  
      graph rnn-cell-1 (
        %H_tminus1[FLOAT, tensor]
        %X_t[FLOAT, tensor]
      ) {
        %Wi = ...
        %Ri = ...
        %Wbi = ...
        %Rbi = ...
        %t1 = X_t * (Wi^T)
        %t2 = H_tminus1*(Ri^T)
        %t3 = Add(%t1, %t2)
        %t4 = Add(%t3, %Wbi)
        %t5 = Add(%t4, %Rbi)
        %Ht = Tanh(%t5)
        %Accumulate = Identity(%Ht)
        return %Ht, %Accumulate
      }
  """

  def __init__(self, initial_state_and_scan_inputs,
    body=None, 
    num_scan_inputs=None, 
    scan_input_axes=None, 
    scan_input_directions=None, 
    scan_output_axes=None, 
    scan_output_directions=None):
    super().__init__('Scan', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      initial_state_and_scan_inputs,
      body=ONNXAttr(body, AttrType.GRAPH), 
      num_scan_inputs=ONNXAttr(num_scan_inputs, AttrType.INT), 
      scan_input_axes=ONNXAttr(scan_input_axes, AttrType.INTS), 
      scan_input_directions=ONNXAttr(scan_input_directions, AttrType.INTS), 
      scan_output_axes=ONNXAttr(scan_output_axes, AttrType.INTS), 
      scan_output_directions=ONNXAttr(scan_output_directions, AttrType.INTS))

class Scatter(ONNXOp):
  """
  Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
  into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
  For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
  for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
  data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
  where i and j are loop counters from 0 up to the respective size in `updates` - 1.
  Example 1:
    data = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    indices = [
        [1, 0, 2],
        [0, 2, 1],
    ]
    updates = [
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
    ]
    output = [
        [2.0, 1.1, 0.0]
        [1.0, 0.0, 2.2]
        [0.0, 2.1, 1.2]
    ]
  Example 2:
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    indices = [[1, 3]]
    updates = [[1.1, 2.1]]
    axis = 1
    output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
  """

  def __init__(self, data, indices, updates,
    axis=None):
    super().__init__('Scatter', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kLong', 'at::kInt'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      data,indices,updates,
      axis=ONNXAttr(axis, AttrType.INT))

class ScatterElements(ONNXOp):
  """
  ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0). The output of the operation
  is produced by creating a copy of the input `data`, and then updating its value
  to values specified by `updates` at specific index positions specified by
  `indices`. Its output shape is the same as the shape of `data`.
  
  For each entry in `updates`, the target index in `data` is obtained by combining
  the corresponding entry in `indices` with the index of the entry itself: the
  index-value for dimension = axis is obtained from the value of the corresponding
  entry in `indices` and the index-value for dimension != axis is obtained from the
  index of the entry itself.
  
  For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
  is performed as below:
  ```
    output[indices[i][j]][j] = updates[i][j] if axis = 0,
    output[i][indices[i][j]] = updates[i][j] if axis = 1,
  ```
  
  This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.
  
  Example 1:
  ```
    data = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    indices = [
        [1, 0, 2],
        [0, 2, 1],
    ]
    updates = [
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
    ]
    output = [
        [2.0, 1.1, 0.0]
        [1.0, 0.0, 2.2]
        [0.0, 2.1, 1.2]
    ]
  ```
  Example 2:
  ```
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    indices = [[1, 3]]
    updates = [[1.1, 2.1]]
    axis = 1
    output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
  ```
  """

  def __init__(self, data, indices, updates,
    axis=None):
    super().__init__('ScatterElements', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong', 'at::kInt'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data,indices,updates,
      axis=ONNXAttr(axis, AttrType.INT))

class ScatterND(ONNXOp):
  """
  ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
  is produced by creating a copy of the input `data`, and then updating its value to values
  specified by `updates` at specific index positions specified by `indices`. Its output shape
  is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
  That is, two or more `updates` for the same index-location is not supported.
  
  `indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
   `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
  Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
  update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
  update to a slice of the tensor.
  
  `updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
  first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
  The remaining dimensions of `updates` correspond to the dimensions of the
  replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
  corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
  must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
  of shapes.
  
  The `output` is calculated via the following equation:
  
      output = np.copy(data)
      update_indices = indices.shape[:-1]
      for idx in np.ndindex(update_indices):
          output[indices[idx]] = updates[idx]
  
  The order of iteration in the above loop is not specified.
  In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
  This ensures that the output value does not depend on the iteration order.
  
  This operator is the inverse of GatherND.
  
  Example 1:
  ```
    data    = [1, 2, 3, 4, 5, 6, 7, 8]
    indices = [[4], [3], [1], [7]]
    updates = [9, 10, 11, 12]
    output  = [1, 11, 3, 10, 9, 6, 7, 12]
  ```
  
  Example 2:
  ```
    data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
               [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
               [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
               [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
    indices = [[0], [2]]
    updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
               [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
    output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
               [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
               [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
               [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  ```
  """

  def __init__(self, data, indices, updates):
    super().__init__('ScatterND', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data,indices,updates)

class Selu(ONNXOp):
  """
  Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  """

  def __init__(self, X,
    alpha=None, 
    gamma=None):
    super().__init__('Selu', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT), 
      gamma=ONNXAttr(gamma, AttrType.FLOAT))

class SequenceAt(ONNXOp):
  """
  Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
  Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
  Negative value means counting positions from the back.
  """

  def __init__(self, input_sequence, position):
    super().__init__('SequenceAt', 1,
      [set(), {'at::kLong', 'at::kInt'}],
      input_sequence,position)

class SequenceConstruct(ONNXOp):
  """
  Construct a tensor sequence containing 'inputs' tensors.
  All tensors in 'inputs' must have the same data type.
  """

  def __init__(self, inputs):
    super().__init__('SequenceConstruct', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      inputs)

class SequenceEmpty(ONNXOp):
  """
  Construct an empty tensor sequence, with given data type.
  """

  def __init__(self,
    dtype=None):
    super().__init__('SequenceEmpty', 1,
      [],
      dtype=ONNXAttr(dtype, AttrType.INT))

class SequenceErase(ONNXOp):
  """
  Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
  Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
  Negative value means counting positions from the back.
  'position' is optional, by default it erases the last tensor from 'input_sequence'.
  """

  def __init__(self, input_sequence, position):
    super().__init__('SequenceErase', 1,
      [set(), {'at::kLong', 'at::kInt'}],
      input_sequence,position)

class SequenceInsert(ONNXOp):
  """
  Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
  'tensor' must have the same data type as 'input_sequence'.
  Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
  Negative value means counting positions from the back.
  'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
  """

  def __init__(self, input_sequence, tensor, position):
    super().__init__('SequenceInsert', 1,
      [set(), {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kLong', 'at::kInt'}],
      input_sequence,tensor,position)

class SequenceLength(ONNXOp):
  """
  Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
  """

  def __init__(self, input_sequence):
    super().__init__('SequenceLength', 1,
      [set()],
      input_sequence)

class Shape(ONNXOp):
  """
  Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  """

  def __init__(self, data):
    super().__init__('Shape', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data)

class Shrink(ONNXOp):
  """
  Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
  having same datatype and shape with input. It has two attributes, lambd and
  bias. The formula of this operator is: If x < -lambd, y = x + bias;
  If x > lambd, y = x - bias; Otherwise, y = 0.
  """

  def __init__(self, input,
    bias=None, 
    lambd=None):
    super().__init__('Shrink', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}],
      input,
      bias=ONNXAttr(bias, AttrType.FLOAT), 
      lambd=ONNXAttr(lambd, AttrType.FLOAT))

class Sigmoid(ONNXOp):
  """
  Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Sigmoid', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class Sign(ONNXOp):
  """
  Calculate the sign of the given input tensor element-wise.
  If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
  """

  def __init__(self, input):
    super().__init__('Sign', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      input)

class Sin(ONNXOp):
  """
  Calculates the sine of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Sin', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Sinh(ONNXOp):
  """
  Calculates the hyperbolic sine of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Sinh', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Size(ONNXOp):
  """
  Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  """

  def __init__(self, data):
    super().__init__('Size', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data)

class Slice(ONNXOp):
  """
  Produces a slice of the input tensor along multiple axes. Similar to numpy:
  https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
  dimension and step for each axis in the list of axes, it uses this information to
  slice the input `data` tensor. If a negative value is passed for any of the
  start or end indices, it represents number of elements before the end of that
  dimension. If the value passed to start or end is larger than the `n` (the
  number of elements in this dimension), it represents `n`. For slicing to the
  end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
  when sclicing forward and 'INT_MIN' when slicing backward.
  If a negative value is passed for step, it represents slicing backward.
  However step value cannot be 0.
  If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
  If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
  Example 1:
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    steps = [1, 2]
    result = [
        [5, 7],
    ]
  Example 2:
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
    ]
  """

  def __init__(self, data, starts, ends, axes, steps):
    super().__init__('Slice', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong', 'at::kInt'}, {'at::kLong', 'at::kInt'}, {'at::kLong', 'at::kInt'}, {'at::kLong', 'at::kInt'}],
      data,starts,ends,axes,steps)

class Softmax(ONNXOp):
  """
  The operator computes the normalized exponential values for the given input:
  
   Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 
  
  The input does not need to explicitly be a 2D vector. The "axis" attribute
  indicates the dimension along which Softmax will be performed.
  The output tensor has the same shape
  and contains the Softmax values of the corresponding input.
  """

  def __init__(self, input,
    axis=None):
    super().__init__('Softmax', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input,
      axis=ONNXAttr(axis, AttrType.INT))

class SoftmaxCrossEntropyLoss(ONNXOp):
  """
  Loss function that measures the softmax cross entropy
  between 'scores' and 'labels'.
  This operator first computes a loss tensor whose shape is identical to the labels input.
  If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
  If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
  the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
  After L is available, this operator can optionally do a reduction operator.
  
  shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
          with K >= 1 in case of K-dimensional loss.
  shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
          with K >= 1 in case of K-dimensional loss.
  
  The loss for one sample, l_i, can caculated as follows:
      l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
  or
      l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
  
  loss is zero for the case when label-value equals ignore_index.
      l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
  
  where:
      p = Softmax(scores)
      y = Log(p)
      c = labels[i][d1][d2]...[dk]
  
  Finally, L is optionally reduced:
  If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
  If reduction = 'sum', the output is scalar: Sum(L).
  If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
  where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
  """

  def __init__(self, scores, labels, weights,
    ignore_index=None, 
    reduction=None):
    super().__init__('SoftmaxCrossEntropyLoss', 2,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}, {'at::kLong', 'at::kInt'}, {'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      scores,labels,weights,
      ignore_index=ONNXAttr(ignore_index, AttrType.INT), 
      reduction=ONNXAttr(reduction, AttrType.STRING))

class Softplus(ONNXOp):
  """
  Softplus takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
  the tensor elementwise.
  """

  def __init__(self, X):
    super().__init__('Softplus', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X)

class Softsign(ONNXOp):
  """
  Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Softsign', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class SpaceToDepth(ONNXOp):
  """
  SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
  this op outputs a copy of the input tensor where values from the height and width dimensions
  are moved to the depth dimension.
  """

  def __init__(self, input,
    blocksize=None):
    super().__init__('SpaceToDepth', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      input,
      blocksize=ONNXAttr(blocksize, AttrType.INT))

class Split(ONNXOp):
  """
  Split a tensor into a list of tensors, along the specified
  'axis'. Lengths of the parts can be specified using input 'split'.
  Otherwise, the tensor is split to equal sized parts.
  """

  def __init__(self, input, split,
    axis=None):
    super().__init__('Split', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      input,split,
      axis=ONNXAttr(axis, AttrType.INT))

class SplitToSequence(ONNXOp):
  """
  Split a tensor into a sequence of tensors, along the specified
  'axis'. Lengths of the parts can be specified using argument 'split'.
  'split' must contain only positive numbers.
  'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
  If 'split' is a scalar, then 'input' will be split into equally sized chunks(if possible).
  Last chunk will be smaller if the 'input' size along the given axis 'axis' is not divisible
  by 'split'.
  Otherwise, the tensor is split into 'size(split)' chunks, with lengths of the parts on 'axis'
  specified in 'split'. In this scenario, the sum of entries in 'split' must be equal to the
  dimension size of input tensor on 'axis'.
  """

  def __init__(self, input, split,
    axis=None, 
    keepdims=None):
    super().__init__('SplitToSequence', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kLong', 'at::kInt'}],
      input,split,
      axis=ONNXAttr(axis, AttrType.INT), 
      keepdims=ONNXAttr(keepdims, AttrType.INT))

class Sqrt(ONNXOp):
  """
  Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.5, is applied to
  the tensor elementwise. If x is negative, then it will return NaN.
  """

  def __init__(self, X):
    super().__init__('Sqrt', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      X)

class Squeeze(ONNXOp):
  """
  Remove single-dimensional entries from the shape of a tensor.
  Takes an input `axes` with a list of axes to squeeze.
  If `axes` is not provided, all the single dimensions will be removed from
  the shape. If an axis is selected with shape entry not equal to one, an error is raised.
  """

  def __init__(self, data, axes):
    super().__init__('Squeeze', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      data,axes)

class StringNormalizer(ONNXOp):
  """
  StringNormalization performs string operations for basic cleaning.
  This operator has only one input (denoted by X) and only one output
  (denoted by Y). This operator first examines the elements in the X,
  and removes elements specified in "stopwords" attribute.
  After removing stop words, the intermediate result can be further lowercased,
  uppercased, or just returned depending the "case_change_action" attribute.
  This operator only accepts [C]- and [1, C]-tensor.
  If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
  if input shape is [C] and shape [1, 1] if input shape is [1, C].
  """

  def __init__(self, X,
    case_change_action=None, 
    is_case_sensitive=None, 
    locale=None, 
    stopwords=None):
    super().__init__('StringNormalizer', 1,
      [set()],
      X,
      case_change_action=ONNXAttr(case_change_action, AttrType.STRING), 
      is_case_sensitive=ONNXAttr(is_case_sensitive, AttrType.INT), 
      locale=ONNXAttr(locale, AttrType.STRING), 
      stopwords=ONNXAttr(stopwords, AttrType.STRINGS))

class Sub(ONNXOp):
  """
  Performs element-wise binary subtraction (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  """

  def __init__(self, A, B):
    super().__init__('Sub', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat', 'at::kBFloat16'}],
      A,B)

class Sum(ONNXOp):
  """
  Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, data_0):
    super().__init__('Sum', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      data_0)

class SVMClassifier(ONNXOp):
  """
      Support Vector Machine classifier
  """

  def __init__(self, X,
    classlabels_ints=None, 
    classlabels_strings=None, 
    coefficients=None, 
    kernel_params=None, 
    kernel_type=None, 
    post_transform=None, 
    prob_a=None, 
    prob_b=None, 
    rho=None, 
    support_vectors=None, 
    vectors_per_class=None):
    super().__init__('SVMClassifier', 2,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      classlabels_ints=ONNXAttr(classlabels_ints, AttrType.INTS), 
      classlabels_strings=ONNXAttr(classlabels_strings, AttrType.STRINGS), 
      coefficients=ONNXAttr(coefficients, AttrType.FLOATS), 
      kernel_params=ONNXAttr(kernel_params, AttrType.FLOATS), 
      kernel_type=ONNXAttr(kernel_type, AttrType.STRING), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING), 
      prob_a=ONNXAttr(prob_a, AttrType.FLOATS), 
      prob_b=ONNXAttr(prob_b, AttrType.FLOATS), 
      rho=ONNXAttr(rho, AttrType.FLOATS), 
      support_vectors=ONNXAttr(support_vectors, AttrType.FLOATS), 
      vectors_per_class=ONNXAttr(vectors_per_class, AttrType.INTS))

class SVMRegressor(ONNXOp):
  """
      Support Vector Machine regression prediction and one-class SVM anomaly detection.
  """

  def __init__(self, X,
    coefficients=None, 
    kernel_params=None, 
    kernel_type=None, 
    n_supports=None, 
    one_class=None, 
    post_transform=None, 
    rho=None, 
    support_vectors=None):
    super().__init__('SVMRegressor', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      coefficients=ONNXAttr(coefficients, AttrType.FLOATS), 
      kernel_params=ONNXAttr(kernel_params, AttrType.FLOATS), 
      kernel_type=ONNXAttr(kernel_type, AttrType.STRING), 
      n_supports=ONNXAttr(n_supports, AttrType.INT), 
      one_class=ONNXAttr(one_class, AttrType.INT), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING), 
      rho=ONNXAttr(rho, AttrType.FLOATS), 
      support_vectors=ONNXAttr(support_vectors, AttrType.FLOATS))

class Tan(ONNXOp):
  """
  Calculates the tangent of the given input tensor, element-wise.
  """

  def __init__(self, input):
    super().__init__('Tan', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      input)

class Tanh(ONNXOp):
  """
  Calculates the hyperbolic tangent of the given input tensor element-wise.
  """

  def __init__(self, input):
    super().__init__('Tanh', 1,
      [{'at::kDouble', 'at::kBFloat16', 'at::kHalf', 'at::kFloat'}],
      input)

class TfIdfVectorizer(ONNXOp):
  """
  This transform extracts n-grams from the input sequence and save them as a vector. Input can
  be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
  For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
  More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
  If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.
  
  In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
  sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
  If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
  Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
  The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
  If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
  indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.
  
  The output vector (denoted by Y) stores the count of each n-gram;
  Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
  between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
  ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
  respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
  Note that we may consider all skips up to S when generating the n-grams.
  
  The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
  the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
  this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.
  
  Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
  If pool_strings is set, the input must be a string tensor.
  """

  def __init__(self, X,
    max_gram_length=None, 
    max_skip_count=None, 
    min_gram_length=None, 
    mode=None, 
    ngram_counts=None, 
    ngram_indexes=None, 
    pool_int64s=None, 
    pool_strings=None, 
    weights=None):
    super().__init__('TfIdfVectorizer', 1,
      [{'at::kLong', 'at::kInt'}],
      X,
      max_gram_length=ONNXAttr(max_gram_length, AttrType.INT), 
      max_skip_count=ONNXAttr(max_skip_count, AttrType.INT), 
      min_gram_length=ONNXAttr(min_gram_length, AttrType.INT), 
      mode=ONNXAttr(mode, AttrType.STRING), 
      ngram_counts=ONNXAttr(ngram_counts, AttrType.INTS), 
      ngram_indexes=ONNXAttr(ngram_indexes, AttrType.INTS), 
      pool_int64s=ONNXAttr(pool_int64s, AttrType.INTS), 
      pool_strings=ONNXAttr(pool_strings, AttrType.STRINGS), 
      weights=ONNXAttr(weights, AttrType.FLOATS))

class ThresholdedRelu(ONNXOp):
  """
  ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
  is applied to the tensor elementwise.
  """

  def __init__(self, X,
    alpha=None):
    super().__init__('ThresholdedRelu', 1,
      [{'at::kDouble', 'at::kHalf', 'at::kFloat'}],
      X,
      alpha=ONNXAttr(alpha, AttrType.FLOAT))

class Tile(ONNXOp):
  """
  Constructs a tensor by tiling a given tensor.
  This is the same as function `tile` in Numpy, but no broadcast.
  For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
  """

  def __init__(self, input, repeats):
    super().__init__('Tile', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      input,repeats)

class TopK(ONNXOp):
  """
  Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
  shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
    -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
      which contains the values of the top k elements along the specified axis
    -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
     contains the indices of the top k elements (original indices from the input
     tensor).
  
  If "largest" is 1 (the default value) then the k largest elements are returned.
  If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
  If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.
  
  Given two equivalent values, this operator uses the indices along the axis as
   a tiebreaker. That is, the element with the lower index will appear first.
  """

  def __init__(self, X, K,
    axis=None, 
    largest=None, 
    sorted=None):
    super().__init__('TopK', 2,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kFloat'}, {'at::kLong'}],
      X,K,
      axis=ONNXAttr(axis, AttrType.INT), 
      largest=ONNXAttr(largest, AttrType.INT), 
      sorted=ONNXAttr(sorted, AttrType.INT))

class Transpose(ONNXOp):
  """
  Transpose the input tensor similar to numpy.transpose. For example, when
  perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
  will be (2, 1, 3).
  """

  def __init__(self, data,
    perm=None):
    super().__init__('Transpose', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}],
      data,
      perm=ONNXAttr(perm, AttrType.INTS))

class TreeEnsembleClassifier(ONNXOp):
  """
      Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
      The attributes named 'nodes_X' form a sequence of tuples, associated by
      index into the sequences, which must all be of equal length. These tuples
      define the nodes.<br>
      Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
      A leaf may have multiple votes, where each vote is weighted by
      the associated class_weights index.<br>
      One and only one of classlabels_strings or classlabels_int64s
      will be defined. The class_ids are indices into this list.
  """

  def __init__(self, X,
    base_values=None, 
    class_ids=None, 
    class_nodeids=None, 
    class_treeids=None, 
    class_weights=None, 
    classlabels_int64s=None, 
    classlabels_strings=None, 
    nodes_falsenodeids=None, 
    nodes_featureids=None, 
    nodes_hitrates=None, 
    nodes_missing_value_tracks_true=None, 
    nodes_modes=None, 
    nodes_nodeids=None, 
    nodes_treeids=None, 
    nodes_truenodeids=None, 
    nodes_values=None, 
    post_transform=None):
    super().__init__('TreeEnsembleClassifier', 2,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      base_values=ONNXAttr(base_values, AttrType.FLOATS), 
      class_ids=ONNXAttr(class_ids, AttrType.INTS), 
      class_nodeids=ONNXAttr(class_nodeids, AttrType.INTS), 
      class_treeids=ONNXAttr(class_treeids, AttrType.INTS), 
      class_weights=ONNXAttr(class_weights, AttrType.FLOATS), 
      classlabels_int64s=ONNXAttr(classlabels_int64s, AttrType.INTS), 
      classlabels_strings=ONNXAttr(classlabels_strings, AttrType.STRINGS), 
      nodes_falsenodeids=ONNXAttr(nodes_falsenodeids, AttrType.INTS), 
      nodes_featureids=ONNXAttr(nodes_featureids, AttrType.INTS), 
      nodes_hitrates=ONNXAttr(nodes_hitrates, AttrType.FLOATS), 
      nodes_missing_value_tracks_true=ONNXAttr(nodes_missing_value_tracks_true, AttrType.INTS), 
      nodes_modes=ONNXAttr(nodes_modes, AttrType.STRINGS), 
      nodes_nodeids=ONNXAttr(nodes_nodeids, AttrType.INTS), 
      nodes_treeids=ONNXAttr(nodes_treeids, AttrType.INTS), 
      nodes_truenodeids=ONNXAttr(nodes_truenodeids, AttrType.INTS), 
      nodes_values=ONNXAttr(nodes_values, AttrType.FLOATS), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING))

class TreeEnsembleRegressor(ONNXOp):
  """
      Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
      All args with nodes_ are fields of a tuple of tree nodes, and
      it is assumed they are the same length, and an index i will decode the
      tuple across these inputs.  Each node id can appear only once
      for each tree id.<br>
      All fields prefixed with target_ are tuples of votes at the leaves.<br>
      A leaf may have multiple votes, where each vote is weighted by
      the associated target_weights index.<br>
      All trees must have their node ids start at 0 and increment by 1.<br>
      Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
  """

  def __init__(self, X,
    aggregate_function=None, 
    base_values=None, 
    n_targets=None, 
    nodes_falsenodeids=None, 
    nodes_featureids=None, 
    nodes_hitrates=None, 
    nodes_missing_value_tracks_true=None, 
    nodes_modes=None, 
    nodes_nodeids=None, 
    nodes_treeids=None, 
    nodes_truenodeids=None, 
    nodes_values=None, 
    post_transform=None, 
    target_ids=None, 
    target_nodeids=None, 
    target_treeids=None, 
    target_weights=None):
    super().__init__('TreeEnsembleRegressor', 1,
      [{'at::kDouble', 'at::kLong', 'at::kInt', 'at::kFloat'}],
      X,
      aggregate_function=ONNXAttr(aggregate_function, AttrType.STRING), 
      base_values=ONNXAttr(base_values, AttrType.FLOATS), 
      n_targets=ONNXAttr(n_targets, AttrType.INT), 
      nodes_falsenodeids=ONNXAttr(nodes_falsenodeids, AttrType.INTS), 
      nodes_featureids=ONNXAttr(nodes_featureids, AttrType.INTS), 
      nodes_hitrates=ONNXAttr(nodes_hitrates, AttrType.FLOATS), 
      nodes_missing_value_tracks_true=ONNXAttr(nodes_missing_value_tracks_true, AttrType.INTS), 
      nodes_modes=ONNXAttr(nodes_modes, AttrType.STRINGS), 
      nodes_nodeids=ONNXAttr(nodes_nodeids, AttrType.INTS), 
      nodes_treeids=ONNXAttr(nodes_treeids, AttrType.INTS), 
      nodes_truenodeids=ONNXAttr(nodes_truenodeids, AttrType.INTS), 
      nodes_values=ONNXAttr(nodes_values, AttrType.FLOATS), 
      post_transform=ONNXAttr(post_transform, AttrType.STRING), 
      target_ids=ONNXAttr(target_ids, AttrType.INTS), 
      target_nodeids=ONNXAttr(target_nodeids, AttrType.INTS), 
      target_treeids=ONNXAttr(target_treeids, AttrType.INTS), 
      target_weights=ONNXAttr(target_weights, AttrType.FLOATS))

class Trilu(ONNXOp):
  """
  Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
  The attribute "upper" determines whether the upper or lower part is retained. If set to true,
  the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
  Default value for the "upper" attribute is true.
  Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
  of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
  All other elements in the matrix are set to zero.
  If k = 0, the triangular part on and above/below the main diagonal is retained.
  If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
  A negative k value retains the main diagonal and |k| diagonals below it.
  If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
  A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
  """

  def __init__(self, input, k,
    upper=None):
    super().__init__('Trilu', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      input,k,
      upper=ONNXAttr(upper, AttrType.INT))

class Unique(ONNXOp):
  """
  Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
  Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
  
  This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
  The first output tensor 'Y' contains all unique values or subtensors of the input.
  The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'..
  The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. ".
  The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.
  
  Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.
  
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
  
  Example 1:
    input_X = [2, 1, 1, 3, 4, 3]
    attribute_sorted = 0
    attribute_axis = None
    output_Y = [2, 1, 3, 4]
    output_indices = [0, 1, 3, 4]
    output_inverse_indices = [0, 1, 1, 2, 3, 2]
    output_counts = [1, 2, 2, 1]
  
  Example 2:
    input_X = [[1, 3], [2, 3]]
    attribute_sorted = 1
    attribute_axis = None
    output_Y = [1, 2, 3]
    output_indices = [0, 2, 1]
    output_inverse_indices = [0, 2, 1, 2]
    output_counts = [1, 1, 2]
  
  Example 3:
    input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
    attribute_sorted = 1
    attribute_axis = 0
    output_Y = [[1, 0, 0], [2, 3, 4]]
    output_indices = [0, 2]
    output_inverse_indices = [0, 0, 1]
    output_counts = [2, 1]
  
  Example 4:
    input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
               [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
    attribute_sorted = 1
    attribute_axis = 1
  
    intermediate data are presented below for better understanding:
  
    there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
    A: [[1, 1], [1, 1]],
       [[0, 1], [0, 1]],
       [[2, 1], [2, 1]],
       [[0, 1], [0, 1]].
  
    there are 3 unique subtensors:
    [[1, 1], [1, 1]],
    [[0, 1], [0, 1]],
    [[2, 1], [2, 1]].
  
    sorted unique subtensors:
    B: [[0, 1], [0, 1]],
       [[1, 1], [1, 1]],
       [[2, 1], [2, 1]].
  
    output_Y is constructed from B:
    [[[0. 1.], [1. 1.], [2. 1.]],
     [[0. 1.], [1. 1.], [2. 1.]]]
  
    output_indices is to map from B to A:
    [1, 0, 2]
  
    output_inverse_indices is to map from A to B:
    [1, 0, 2, 0]
  
    output_counts = [2 1 1]
  """

  def __init__(self, X,
    axis=None, 
    sorted=None):
    super().__init__('Unique', 4,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      X,
      axis=ONNXAttr(axis, AttrType.INT), 
      sorted=ONNXAttr(sorted, AttrType.INT))

class Unsqueeze(ONNXOp):
  """
  Insert single-dimensional entries to the shape of an input tensor (`data`).
  Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).
  
  For example:
    Given an input tensor (`data`) of shape [3, 4, 5], then
    Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].
  
  The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
  The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
  Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
  The order of values in `axes` does not matter and can come in any order.
  """

  def __init__(self, data, axes):
    super().__init__('Unsqueeze', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat', 'at::kBFloat16'}, {'at::kLong'}],
      data,axes)

class Upsample(ONNXOp):
  """
  Upsample the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * scale).
  """

  def __init__(self, X, scales,
    mode=None):
    super().__init__('Upsample', 1,
      [{'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kFloat'}],
      X,scales,
      mode=ONNXAttr(mode, AttrType.STRING))

class Where(ONNXOp):
  """
      Return elements, either from X or Y, depending on condition
      (with Numpy-style broadcasting support).
      Where behaves like numpy.where with three parameters:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
  """

  def __init__(self, condition, X, Y):
    super().__init__('Where', 1,
      [{'at::kBool'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}, {'at::kDouble', 'at::kLong', 'at::kByte', 'at::kInt', 'at::kHalf', 'at::kShort', 'at::kBool', 'at::kFloat'}],
      condition,X,Y)

class Xor(ONNXOp):
  """
  Returns the tensor resulted from performing the `xor` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  """

  def __init__(self, A, B):
    super().__init__('Xor', 1,
      [{'at::kBool'}, {'at::kBool'}],
      A,B)

class ZipMap(ONNXOp):
  """
      Creates a map from the input and the attributes.<br>
      The values are provided by the input tensor, while the keys are specified by the attributes.
      Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
      The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
  """

  def __init__(self, X,
    classlabels_int64s=None, 
    classlabels_strings=None):
    super().__init__('ZipMap', 1,
      [{'at::kFloat'}],
      X,
      classlabels_int64s=ONNXAttr(classlabels_int64s, AttrType.INTS), 
      classlabels_strings=ONNXAttr(classlabels_strings, AttrType.STRINGS))

onnx_ops = {
  'adam': Adam,
  'adagrad': Adagrad,
  'momentum': Momentum,
  'gradient': Gradient,
  'zipmap': ZipMap,
  'onehotencoder': OneHotEncoder,
  'normalizer': Normalizer,
  'linearclassifier': LinearClassifier,
  'labelencoder': LabelEncoder,
  'imputer': Imputer,
  'featurevectorizer': FeatureVectorizer,
  'treeensembleregressor': TreeEnsembleRegressor,
  'dictvectorizer': DictVectorizer,
  'castmap': CastMap,
  'shape': Shape,
  'reshape': Reshape,
  'binarizer': Binarizer,
  'reciprocal': Reciprocal,
  'leakyrelu': LeakyRelu,
  'hardsigmoid': HardSigmoid,
  'treeensembleclassifier': TreeEnsembleClassifier,
  'reducemin': ReduceMin,
  'div': Div,
  'randomnormallike': RandomNormalLike,
  'randomnormal': RandomNormal,
  'greaterorequal': GreaterOrEqual,
  'pow': Pow,
  'or': Or,
  'mul': Mul,
  'min': Min,
  'floor': Floor,
  'mean': Mean,
  'lrn': LRN,
  'scaler': Scaler,
  'max': Max,
  'round': Round,
  'lppool': LpPool,
  'sigmoid': Sigmoid,
  'relu': Relu,
  'quantizelinear': QuantizeLinear,
  'logsoftmax': LogSoftmax,
  'randomuniform': RandomUniform,
  'depthtospace': DepthToSpace,
  'concat': Concat,
  'bitshift': BitShift,
  'ceil': Ceil,
  'gather': Gather,
  'log': Log,
  'reducesumsquare': ReduceSumSquare,
  'dropout': Dropout,
  'greater': Greater,
  'reducesum': ReduceSum,
  'sequenceempty': SequenceEmpty,
  'neg': Neg,
  'constant': Constant,
  'maxpool': MaxPool,
  'sub': Sub,
  'reducelogsumexp': ReduceLogSumExp,
  'xor': Xor,
  'globallppool': GlobalLpPool,
  'upsample': Upsample,
  'prelu': PRelu,
  'loop': Loop,
  'lpnormalization': LpNormalization,
  'dynamicquantizelinear': DynamicQuantizeLinear,
  'splittosequence': SplitToSequence,
  'linearregressor': LinearRegressor,
  'add': Add,
  'selu': Selu,
  'reducemax': ReduceMax,
  'and': And,
  'abs': Abs,
  'qlinearmatmul': QLinearMatMul,
  'lessorequal': LessOrEqual,
  'clip': Clip,
  'argmax': ArgMax,
  'einsum': Einsum,
  'hardmax': Hardmax,
  'conv': Conv,
  'globalmaxpool': GlobalMaxPool,
  'maxunpool': MaxUnpool,
  'argmin': ArgMin,
  'averagepool': AveragePool,
  'sqrt': Sqrt,
  'size': Size,
  'instancenormalization': InstanceNormalization,
  'gemm': Gemm,
  'reducelogsum': ReduceLogSum,
  'cos': Cos,
  'not': Not,
  'eyelike': EyeLike,
  'equal': Equal,
  'cast': Cast,
  'exp': Exp,
  'flatten': Flatten,
  'svmclassifier': SVMClassifier,
  'roialign': RoiAlign,
  'reducemean': ReduceMean,
  'scatter': Scatter,
  'split': Split,
  'identity': Identity,
  'reducel2': ReduceL2,
  'globalaveragepool': GlobalAveragePool,
  'tan': Tan,
  'reducel1': ReduceL1,
  'lstm': LSTM,
  'slice': Slice,
  'softmax': Softmax,
  'softmaxcrossentropyloss': SoftmaxCrossEntropyLoss,
  'categorymapper': CategoryMapper,
  'maxroipool': MaxRoiPool,
  'softsign': Softsign,
  'gathernd': GatherND,
  'batchnormalization': BatchNormalization,
  'spacetodepth': SpaceToDepth,
  'squeeze': Squeeze,
  'unique': Unique,
  'sum': Sum,
  'sinh': Sinh,
  'less': Less,
  'tanh': Tanh,
  'isnan': IsNaN,
  'tile': Tile,
  'multinomial': Multinomial,
  'topk': TopK,
  'reversesequence': ReverseSequence,
  'transpose': Transpose,
  'stringnormalizer': StringNormalizer,
  'acos': Acos,
  'asin': Asin,
  'gru': GRU,
  'atan': Atan,
  'sign': Sign,
  'trilu': Trilu,
  'where': Where,
  'sin': Sin,
  'shrink': Shrink,
  'matmul': MatMul,
  'expand': Expand,
  'scan': Scan,
  'compress': Compress,
  'elu': Elu,
  'unsqueeze': Unsqueeze,
  'constantofshape': ConstantOfShape,
  'onehot': OneHot,
  'sequenceat': SequenceAt,
  'cosh': Cosh,
  'asinh': Asinh,
  'rnn': RNN,
  'acosh': Acosh,
  'atanh': Atanh,
  'erf': Erf,
  'nonzero': NonZero,
  'meanvariancenormalization': MeanVarianceNormalization,
  'scatternd': ScatterND,
  'randomuniformlike': RandomUniformLike,
  'resize': Resize,
  'mod': Mod,
  'thresholdedrelu': ThresholdedRelu,
  'matmulinteger': MatMulInteger,
  'pad': Pad,
  'convinteger': ConvInteger,
  'qlinearconv': QLinearConv,
  'celu': Celu,
  'convtranspose': ConvTranspose,
  'dequantizelinear': DequantizeLinear,
  'sequencelength': SequenceLength,
  'nonmaxsuppression': NonMaxSuppression,
  'isinf': IsInf,
  'cumsum': CumSum,
  'softplus': Softplus,
  'gatherelements': GatherElements,
  'scatterelements': ScatterElements,
  'range': Range,
  'svmregressor': SVMRegressor,
  'negativeloglikelihoodloss': NegativeLogLikelihoodLoss,
  'det': Det,
  'sequenceconstruct': SequenceConstruct,
  'if': If,
  'sequenceinsert': SequenceInsert,
  'tfidfvectorizer': TfIdfVectorizer,
  'sequenceerase': SequenceErase,
  'concatfromsequence': ConcatFromSequence,
  'hardswish': HardSwish,
  'reduceprod': ReduceProd,
  'arrayfeatureextractor': ArrayFeatureExtractor,
}
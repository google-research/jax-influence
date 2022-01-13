# Copyright 2021 The Jax Influence Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data types."""

from typing import Any, Callable, Generic, List, Mapping, Sequence, Tuple, TypeVar, Union

import flax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# Data type of PRNGKey.
PRNGKey = Any
# Data type of a PyTree.
PyTree = Any
# Data type for a numerical array.
Array = TypeVar('Array', np.ndarray, jnp.ndarray)
# Data type for a scalar.
Scalar = TypeVar('Scalar', np.float32, np.float64, jnp.float32, jnp.float64,
                 float)
# Type of Batch.
Batch = TypeVar('Batch', Mapping[str, Array], Array)
TFBatch = Mapping[str, tf.Tensor]
# Shape of Array.
Shape = Tuple[int]

# Function to select a subset of parameters.
SelectionFn = TypeVar('SelectionFn', bound=Callable[[Sequence[str]], bool])
# Generic function of params. Currently Python does not allow varargs in
# Callable.
Fn = Any
# Scalar-valued function.
ScalarFn = TypeVar('ScalarFn', Callable[[PyTree, Batch], Scalar],
                   Callable[[PyTree], Scalar])
# Array-valued function.
ArrayFn = TypeVar('ArrayFn', Callable[[PyTree, Batch], Union[Array, PyTree]],
                  Callable[[PyTree], Union[Array, PyTree]])
# HVP function.
HVPFn = TypeVar(
    'HVPFn', bound=Callable[[PyTree, PyTree, Batch], Union[Array, PyTree]])

# Type of Jax Client Backend. It cannot be annotated in Python, but it refers
# to tensorflow.compiler.xla.python.xla_extension.Client.
# To get the backend use: client = jax.lib.xla_bridge.get_backend().
JaxClient = Any


# Note use of Generic to bring array in the scope of these dataclasses.
@flax.struct.dataclass
class ArnoldiResult(Generic[Array]):
  """Represents result of Arnoldi Iteration.

  Attributes:
    matrix: Array of dimension (n, n-1) representing the matrix to approximate
      on the Krylov subspace, where n is the number of Arnoldi Iterations.
    proj: Orthogonal projections on the Krylov subspace.
  """
  matrix: Array
  proj: List[PyTree]


@flax.struct.dataclass
class ArnoldiEigens(Generic[Array]):
  """Result of distilling an ArnoldiResult.

  Attributes:
    eigenval: Array of dimension (n,) containing the eigenvalues.
    eigenvec: List of n PyTrees, representing the eigenvector estimates wrt. the
      original base in the ambient space (and not the projections selected as a
      basis for the Krylov subspace by the Arnoldi iteration).
  """
  eigenval: Array
  eigenvec: List[PyTree]

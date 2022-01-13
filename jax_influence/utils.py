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

"""General utilities to use this package."""

from typing import Dict, List, Optional, Tuple

from absl import logging
import flax
import jax
import jax.numpy as jnp
from jax_influence.types import Array
from jax_influence.types import ArrayFn
from jax_influence.types import PyTree
import numpy as np
import tensorflow as tf


def reshape_batch_for_pmap(batch: Dict[str, Array]) -> Dict[str, Array]:
  """Reshapes an input batch for use with pmap.

  Args:
    batch: A batch with arrays.

  Returns:
    The batch with, to each array in it, an extra dimension of `device_count`.
  """

  def _reshape(x):
    # Reshape (host_batch_size, ...) to (num_devices, device_batch_size, ...).
    return x.reshape((jax.local_device_count(), -1) + x.shape[1:])

  return jax.tree_map(_reshape, batch)


def get_eigvals_eigvecs(location: str,
                        eigvals_slice: Optional[Tuple[int, int]] = None,
                        use_abs: bool = True) -> Tuple[Array, List[PyTree]]:
  """Return eigenvectors and eigenvalues loaded from file.

  Args:
   location: a cns file path where the eigenvalues and eigenvectors are stored.
   eigvals_slice: two integer values indicating the indexes of eigenvalues to
     keep after they are sorted in a ascending order. It is recommended to keep
     the largest eigenvalues.
   use_abs: If True uses the absolute value when sorting the eigenvalues,
     otherwise it uses the value.

  Returns:
    Array of eigenvalues in ascending order and a list of corresponding
    eigenvectors (with the tree structure matching the model's parameters).
  """
  with tf.io.gfile.GFile(location, 'rb') as f:
    proj_data = flax.serialization.msgpack_restore(f.read())
  # Older version of the code was using a tuple, new version uses
  # the type ArnoldiEigens.
  if '0' in proj_data:
    key_eigenval = '0'
  else:
    key_eigenval = 'eigenval'
  eigvals = proj_data[key_eigenval]
  if key_eigenval == '0':
    key_eigenvec = '1'
  else:
    key_eigenvec = 'eigenvec'
  if use_abs:
    idx = np.argsort(-np.abs(eigvals))
  else:
    idx = np.argsort(-eigvals)
  eigvals = proj_data[key_eigenval][idx]
  eigvecs_unsorted = [v for _, v in proj_data[key_eigenvec].items()]
  eigvecs = [eigvecs_unsorted[i] for i in idx]
  if eigvals_slice:
    # Slice the eigenvals. Make sure have the top k biggest ones.
    logging.info('Using slice of eigvals %s', eigvals_slice)
    low, high = eigvals_slice
    eigvals = eigvals[low:high]
    eigvecs = eigvecs[low:high]
  eigvecs = [flax.core.FrozenDict(v) for v in eigvecs]
  return eigvals, eigvecs


def get_projection(
    batch: Dict[str, Array],
    batch_size: int,
    jvp_fn: ArrayFn,
    params: PyTree,
    eigvecs: List[PyTree],
    eigvals: Optional[Array] = None,
) -> Array:
  """Return the projections."""
  batch = reshape_batch_for_pmap(batch)
  tangents = jax.tree_map(np.array, eigvecs)
  primal = flax.jax_utils.replicate(params)
  tangents = flax.jax_utils.replicate(tangents)
  jvp_fn = jax.pmap(jvp_fn, axis_name='batch')
  projections = jnp.zeros((batch_size, len(eigvecs)))
  for i, tangent in enumerate(tangents):
    prj = jvp_fn(primal, tangent, batch)
    prj = prj.flatten()
    projections = jax.ops.index_update(projections, jax.ops.index[:, i], prj)
  if eigvals is None:
    return jnp.asarray(projections)
  return jnp.asarray(projections @ jnp.diag(1.0 / eigvals))

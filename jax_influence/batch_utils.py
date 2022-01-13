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

"""Utilities to work with batches."""

from typing import Callable, Iterator, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax_influence.types import Array
from jax_influence.types import Batch
from jax_influence.types import TFBatch
import numpy as np
import tensorflow as tf


def get_first_batch_element(
    batch: Union[Batch, TFBatch]) -> Union[Array, tf.Tensor]:
  """Gets first element of a batch."""
  if isinstance(batch, np.ndarray) or isinstance(batch, jnp.ndarray):
    element = batch
  else:
    element = next(iter(batch.values()))
  return element


def get_microbatch(batch: Batch, idx: int, micro_batch_size: int) -> Batch:
  """Gets a micro batch from a batch to use for accumulation (e.g. gradient)."""
  offset = idx * micro_batch_size
  length = micro_batch_size
  starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
  limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
  return {
      k: jax.lax.dynamic_slice(b, starts[k], limits[k])
      for k, b in batch.items()
  }


def get_batch_size(batch: Batch) -> int:
  """Returns the size of a batch."""
  element = get_first_batch_element(batch)
  return element.shape[0]


def maybe_convert_to_array(batch: Union[Batch, TFBatch],
                           convert_fn=jnp.array) -> Batch:
  """Converts a batch consiting of tf.Tensors to arrays.

  Args:
    batch: The batch to convert.
    convert_fn: The function to convert a tf.Tensor to an array. By default it
      converts to JaX arrays. For Numpy please set convert_fn=np.array.

  Returns:
    The corresponding batch with each tf.Tensor converted to an array.

  """
  if convert_fn not in (jnp.array, np.array):
    raise ValueError('Only Numpy and Jax Array supported')
  element = get_first_batch_element(batch)
  if isinstance(element, tf.Tensor):
    return jax.tree_map(convert_fn, batch)
  return batch


def shard(batch: Batch) -> Batch:
  """Shards a batch across multiple devices."""
  local_device_count = jax.local_device_count()
  return jax.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), batch)


class BatchIterator:
  """An Iterator applying transforms to batches.

  Attributes:
    _data_iter: An Iterator yielding batches.
    _transforms: A Sequence of transformations to apply to each batch yielded by
      data_iter.
  """

  def __init__(self,
               data_iter: Iterator[Union[Batch, TFBatch]],
               transforms: Optional[Sequence[Callable[[Union[Batch, TFBatch]],
                                                      Union[Batch,
                                                            TFBatch]]]] = None):
    self._data_iter = data_iter
    self._transforms = transforms

  def __iter__(self):
    return self

  def __next__(self):
    batch = next(self._data_iter)
    if self._transforms is not None:
      for fn in self._transforms:
        batch = fn(batch)

    return batch

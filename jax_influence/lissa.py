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

"""Implementation of LiSSA.

LiSSA is a method for solving Ax = b by iteration, see
http://arxiv.org/pdf/1602.03943.pdf. It can be used when A is too expensive to
store but evaluating Ax is affordable.

"""
from typing import Callable

from absl import logging
import jax
from jax_influence.types import PyTree


def lissa(matrix_fn: Callable[[PyTree], PyTree],
          vector: PyTree,
          recursion_depth: int,
          scale: float = 10,
          damping: float = 0.0,
          log_progress: bool = False) -> PyTree:
  """Estimates A^{-1}v following the LiSSA algorithm.

  See the paper http://arxiv.org/pdf/1602.03943.pdf.

  [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v

  Args:
    matrix_fn: Function taking a vector v and returning Av.
    vector: The vector v for which we want to compute A^{-1}v.
    recursion_depth: Depth of the LiSSA iteration.
    scale: Rescaling factor for A; the algorithm requires ||A / scale|| < 1.
    damping: Damping factor to regularize a nearly-singular A.
    log_progress: If True reports progress.

  Returns:
    An estimate of A^{-1}v.
  """
  if not damping >= 0.0:
    raise ValueError("Damping factor should be positive.")
  if not scale >= 1.0:
    raise ValueError("Scaling factor should be larger than 1.0.")

  curr_estimate = vector
  for i in range(recursion_depth):
    matrix_vector = matrix_fn(curr_estimate)
    curr_estimate = jax.tree_util.tree_multimap(
        lambda v, u, h: v + (1 - damping) * u - h / scale, vector,
        curr_estimate, matrix_vector)
    if log_progress:
      logging.info("LISSA: %d", i)

  curr_estimate = jax.tree_map(lambda x: x / scale, curr_estimate)
  return curr_estimate

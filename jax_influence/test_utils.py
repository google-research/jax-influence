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

"""Testing utilities."""

import functools
from typing import Optional

from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from jax_influence.types import Array
from jax_influence.types import PyTree
import numpy as np


def _assert_numpy_allclose(a: np.ndarray,
                           b: np.ndarray,
                           atol: Optional[float] = None,
                           rtol: Optional[float] = None,
                           err_msg=''):
  """Helper for asserting closeness of numpy arrays."""
  kw = {}
  if atol:
    kw['atol'] = atol
  if rtol:
    kw['rtol'] = rtol

  np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def _assert_shapes_equal(a: np.ndarray, b: np.ndarray):
  """Helper to assert equality of shapes of numpy arrays."""
  assert a.shape == b.shape, f'Shapes differ: {a.shape} != {b.shape}'


def leaves_to_jndarray(pytree: PyTree):
  """Converts leaves of pytree to jax.numpy arrays."""
  return jax.tree_map(jnp.array, pytree)


def check_equal(a: PyTree, b: PyTree, err_msg: str = ''):
  """Tests equality of a and b."""
  # Conversion required by jax tree_util if a or b are dicts.
  if isinstance(a, dict):
    a = FrozenDict(a)
  if isinstance(b, dict):
    b = FrozenDict(b)
  assert_close = functools.partial(_assert_numpy_allclose, err_msg=err_msg)
  assertions = jax.tree_multimap(assert_close, a, b)

  jax.tree_util.tree_all(assertions)


def check_leaves_have_same_shape(a: PyTree, b: PyTree):
  """Tests that the leaves of a and b have same shape."""
  assertions = jax.tree_multimap(_assert_shapes_equal, a, b)

  jax.tree_util.tree_all(assertions)


def check_close(a: PyTree,
                b: PyTree,
                atol: Optional[float] = None,
                rtol: Optional[float] = None,
                err_msg: str = ''):
  """Tests closeness of a and b."""
  # Conversion required by jax tree_util if a or b are dicts.
  if isinstance(a, dict):
    a = FrozenDict(a)
  if isinstance(b, dict):
    b = FrozenDict(b)
  assert_close = functools.partial(
      _assert_numpy_allclose, atol=atol, rtol=rtol, err_msg=err_msg)
  assertions = jax.tree_multimap(assert_close, a, b)

  jax.tree_util.tree_all(assertions)


def gen_non_hermitian_log_space_eigvals(n: int) -> Array:
  """Generates a random non-hermitian matrix with equally spaced eigenvalues.

  Args:
    n: Dimension of matrix to generate.

  Returns:
    A Random non-hermitian matrix. The eigenvalues are equally
    spaced between 10**-2 and 10**2 in log-space.
  """
  eigenvectors = np.random.normal(size=(n, n))
  desired_eigenvalues = np.array(
      [10**eigenvectors for eigenvectors in np.linspace(-2, 2, num=n)])
  test_matrix = np.linalg.solve(eigenvectors,
                                np.diag(desired_eigenvalues) @ eigenvectors)
  return test_matrix


def gen_hermitian_log_space_eigvals(n: int) -> Array:
  """Generates a random hermitian matrix with equally spaced eigenvalues.

  Args:
    n: Dimension of matrix to generate.

  Returns:
    A Random non-hermitian matrix. The eigenvalues are equally
    spaced between 10**-2 and 10**2 in log-space.
  """
  x = np.random.normal(size=n * n).reshape((n, n))
  q, _ = np.linalg.qr(x, mode='complete')
  desired_eigenvalues = np.array([10**x for x in np.linspace(-2, 2, num=n)])
  test_matrix = q.T @ np.diag(desired_eigenvalues) @ q
  return test_matrix


class InfiniteIterator:
  """Infinite Iterator used for testing.

  Attributes:
    seed: Initial seed for producing batches. It will incremented after each
      iteration.
    batch_fn: A function mapping seeds to batches.
  """

  def __init__(self, seed, batch_fn):
    self.seed = seed
    self.batch_fn = batch_fn

  def __iter__(self):
    return self

  def __next__(self):
    batch = self.batch_fn(self.seed)
    self.seed += 1
    return batch

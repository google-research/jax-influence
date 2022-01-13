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

"""Tests for linalg_utils."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_influence import linalg_utils
from jax_influence import test_utils
import numpy as np


class LinalgUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='dictionary',
          pytree={
              'a': 5,
              'b': 25
          },
          pytree_to_add={
              'a': 1,
              'b': 44
          },
          scalar=22,
          expected={
              'a': 27,
              'b': 993
          }))
  def test_add_scalar_multiple(self, pytree, scalar, pytree_to_add, expected):
    actual = linalg_utils.add_scalar_multiple(pytree, scalar, pytree_to_add)
    test_utils.check_equal(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='jax_backend',
          pytree_a={
              'a': 5.0,
              'nested': {
                  'b': 2.0,
                  'c': np.array([3.0, 1.0]),
              }
          },
          pytree_b={
              'a': 3.0,
              'nested': {
                  'b': 1.0,
                  'c': np.array([7.0, 1.0]),
              }
          },
          use_jax=True,
          expected=39.0,
      ),
      dict(
          testcase_name='numpy_backend',
          pytree_a={
              'a': 5.0,
              'nested': {
                  'b': 2.0,
                  'c': np.array([3.0, 1.0]),
              }
          },
          pytree_b={
              'a': 3.0,
              'nested': {
                  'b': 1.0,
                  'c': np.array([7.0, 1.0]),
              }
          },
          use_jax=False,
          expected=39.0,
      ))
  def test_inner_product(self, pytree_a, pytree_b, use_jax, expected):
    actual = linalg_utils.inner_product(pytree_a, pytree_b, use_jax=use_jax)
    if use_jax:
      self.assertIsInstance(actual, jnp.ndarray)
    else:
      self.assertIsInstance(actual, np.float64)

    test_utils.check_close(actual, expected, atol=1e-6, rtol=1e-6)

  def test_project(self):
    proj1 = {
        'a': np.array([1, -1, 1]),
        'nested': {
            'b': np.array([2, 3]),
            'c': np.array([4, 7])
        }
    }
    proj2 = {
        'a': np.array([1, 0, -1]),
        'nested': {
            'b': np.array([3, -2]),
            'c': np.array([7, -4])
        }
    }
    pytree = {
        'a': np.array([0, 1, 2]),
        'nested': {
            'b': np.array([0, 1]),
            'c': np.array([1, 0])
        }
    }
    expected = np.array([8.0, 3.0])

    actual = linalg_utils.project([proj1, proj2], pytree)
    test_utils.check_close(actual, expected, atol=1e-6, rtol=1e-6)

  def test_decompose_by_projection(self):
    # Projection components orthogonal and of norm sqrt(42.0).
    proj1 = {
        'a': np.array([1, 1]),
        'nested': {
            'b': np.array([2, 2]),
            'c': np.array([4, 4])
        }
    }
    proj2 = {
        'a': np.array([1, -1]),
        'nested': {
            'b': np.array([2, -2]),
            'c': np.array([4, -4])
        }
    }
    proj = [proj1, proj2]
    n_const = jnp.sqrt(42.0)
    # PyTree to project.
    pytree = {
        'a': np.array([0, 1]),
        'nested': {
            'b': np.array([0, 1]),
            'c': np.array([1, 0])
        }
    }

    check_close = functools.partial(
        test_utils.check_close, atol=1e-6, rtol=1e-6)
    # Decompose pytree.
    t_proj, t_perp = linalg_utils.decompose_by_projection(
        proj, pytree, norm_const=n_const)

    # Check orthogonality of t_perp to each element of proj.
    for p in proj:
      check_close(0.0, linalg_utils.inner_product(t_perp, p))
    # Check that reconstruction of proj from t_proj, t_perp is correct.
    t_rec = t_perp
    for i, p in enumerate(proj):
      t_rec = linalg_utils.add_scalar_multiple(t_rec, t_proj[i] / n_const,
                                               proj[i])
    check_close(pytree, t_rec)

  @parameterized.named_parameters(
      dict(testcase_name='case1', seed=0, matrix_shape=(5, 8), proj_size=44),
      dict(testcase_name='case2', seed=5, matrix_shape=(9, 17), proj_size=2),
  )
  def test_compose_projections(self, seed, matrix_shape, proj_size):
    seed = jax.random.PRNGKey(seed)
    seed_mat, seed_proj = jax.random.split(seed)
    mat = jax.random.normal(seed_mat, shape=matrix_shape)

    # Random initialization of proj.
    proj = [jnp.zeros((proj_size,)) for _ in range(matrix_shape[0])]
    proj = linalg_utils.random_initialization(proj, seed_proj)
    # Stack into a matrix P.
    proj_mat = [jnp.expand_dims(p, 0) for p in proj]
    proj_mat = jnp.concatenate(proj_mat, axis=0)
    # Compute expected result.
    expected = jnp.einsum('ik,ij -> jk', proj_mat, mat)
    expected = [expected[i] for i in range(expected.shape[0])]

    result = linalg_utils.change_basis_of_projections(mat, proj)
    test_utils.check_close(result, expected, atol=1e-6, rtol=1e-6)

  def test_random_initialization(self):
    pytree = {
        'a': np.array(5),
        'nested': {
            'b': np.array([2, 3, 4]),
            'c': np.array([1, 2])
        }
    }
    seed = jax.random.PRNGKey(0)
    random_pytree = linalg_utils.random_initialization(pytree, seed)

    # Check structure is the same.
    pstruct = jax.tree_structure(pytree)
    rstruct = jax.tree_structure(random_pytree)

    self.assertEqual(pstruct, rstruct)

    test_utils.check_leaves_have_same_shape(pytree, random_pytree)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for lissa."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_influence import lissa
from jax_influence import test_utils
import numpy as np


class LissaTest(absltest.TestCase):

  def test_no_damping(self):
    n = 10

    # LiSSA does not necessarily work well with generic matrices.
    # Here we take a perturbation of the identity.
    seed = jax.random.PRNGKey(0)
    seed_mat, seed_vec = jax.random.split(seed)
    mat = jax.random.normal(seed_mat, shape=(n, n))
    mat = jnp.eye(n) - 0.3 * mat
    vec = jax.random.normal(seed_vec, shape=(n,))

    result = lissa.lissa(lambda x: mat @ x, vec, recursion_depth=1000, scale=20)
    expected = jnp.linalg.solve(mat, vec)
    test_utils.check_close(result, expected, atol=5e-5, rtol=5e-5)

  def test_w_damping(self):
    n = 10

    # We generate an almost singular matrix and use damping to regularize
    # LiSSA.
    seed = jax.random.PRNGKey(0)
    seed_mat1, seed_mat2, seed_vec = jax.random.split(seed, 3)
    mat1 = jax.random.normal(seed_mat1, shape=(n, n - 1))
    # mat1 gives rise to an orthogonal matrix of rank n-1.
    q, _ = np.linalg.qr(mat1, mode='reduced')
    mat2 = jax.random.normal(seed_mat2, shape=(n, 1))
    # This matrix is ill-conditioned.
    mat = q @ q.T + 1e-20 * mat2 @ mat2.T
    vec = jax.random.normal(seed_vec, shape=(n,))
    result = lissa.lissa(
        lambda x: mat @ x, vec, recursion_depth=100, scale=20, damping=1e-4)

    # We test on the error of solving the linear system.
    err = np.linalg.norm(mat @ result - vec, ord=np.inf)
    self.assertLess(err, 5e-2)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for arnoldi."""

from absl.testing import absltest
from absl.testing import parameterized
from jax_influence import arnoldi
from jax_influence import test_utils
from jax_influence.types import ArnoldiEigens
import numpy as np


class ArnoldiTest(parameterized.TestCase):

  def _eigval_max_error(self, matrix_on_krylov, true_matrix, top_k):
    """Computes max error on eigenvalue approximation of top_k values."""
    n = matrix_on_krylov.shape[1]
    appx = np.sort(np.linalg.eigvals(matrix_on_krylov[:n, :n]))
    eigvals = np.sort(np.linalg.eigvals(true_matrix))
    errors = (appx[-top_k:] - eigvals[-top_k:])
    return np.max(np.abs(errors))

  def _eigval_eigvecs_max_error(self, distilled: ArnoldiEigens, true_matrix,
                                top_k, hermitian: bool):
    """Computes max approximation error on top_k eigenvalues/vectors."""
    if hermitian:
      eigvals = np.linalg.eigvalsh(true_matrix)
    else:
      eigvals = np.linalg.eigvals(true_matrix)
    idx = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvals = eigvals[-top_k:]

    err_val = np.max(np.abs(eigvals - distilled.eigenval))
    # Test directly if distilled eigenvectors are eigenvectors for
    # the corresponding eigenvalues.
    err_vec = [
        true_matrix @ distilled.eigenvec[i] - eigvals[i] * distilled.eigenvec[i]
        for i in range(eigvals.shape[0])
    ]
    err_vec = [np.linalg.norm(x, ord=np.inf) for x in err_vec]
    err_vec = np.max(err_vec)
    return err_vec, err_val

  @parameterized.named_parameters(
      dict(
          testcase_name='non_hermitian',
          hermitian=False,
          norm_const=1.0,
          matrix_size=100,
          n_iters=50),
      dict(
          testcase_name='non_hermitian_rescale',
          hermitian=False,
          norm_const=3.0,
          matrix_size=100,
          n_iters=50),
      dict(
          testcase_name='hermitian',
          hermitian=True,
          norm_const=1.0,
          matrix_size=100,
          n_iters=50),
      dict(
          testcase_name='hermitian_rescale',
          hermitian=True,
          norm_const=3.0,
          matrix_size=100,
          n_iters=50))
  def test_locates_eigenvalues(self, hermitian, norm_const, matrix_size,
                               n_iters):
    if hermitian:
      matrix = test_utils.gen_hermitian_log_space_eigvals(matrix_size)
    else:
      matrix = test_utils.gen_non_hermitian_log_space_eigvals(matrix_size)
    arnoldi_result = arnoldi.arnoldi_iter(
        matrix_fn=lambda v: matrix @ v,
        dot_product_fn=np.dot,
        start_vector=np.random.normal(size=(matrix_size,)),
        n_iters=n_iters,
        norm_constant=norm_const)
    err = self._eigval_max_error(arnoldi_result.matrix, matrix, top_k=10)
    # Test succeeds if error < 5% (top eigenvalues are ~100).
    self.assertLess(err, 5, 'Eigenvalue estimation error is not < 5%')

  @parameterized.named_parameters(
      dict(testcase_name='hermitian', force_hermitian=True),
      dict(testcase_name='non_hermitian', force_hermitian=False),
  )
  def test_distill(self, force_hermitian):
    if force_hermitian:
      matrix = test_utils.gen_hermitian_log_space_eigvals(100)
    else:
      matrix = test_utils.gen_non_hermitian_log_space_eigvals(100)
    arnoldi_result = arnoldi.arnoldi_iter(
        matrix_fn=lambda v: matrix @ v,
        dot_product_fn=np.dot,
        start_vector=np.random.normal(size=(100,)),
        n_iters=50)
    distilled = arnoldi.distill(
        arnoldi_result, force_hermitian=force_hermitian, top_k=10)
    err_val, err_vec = self._eigval_eigvecs_max_error(
        distilled, matrix, top_k=10, hermitian=force_hermitian)
    # Recall that the top 10 eigenvalues are close to 100 by construction.
    self.assertLess(err_val, 6,
                    'Relative eigenvalue estimation error is not < 6%')
    self.assertLess(err_vec, 1e-2, 'Eigenvector estimation error is not < 0.01')

if __name__ == '__main__':
  absltest.main()

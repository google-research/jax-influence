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

"""Implementation of the Arnoldi Iteration."""

from typing import Callable, Optional

from absl import logging
import jax
import jax.numpy as jnp
from jax_influence import linalg_utils
from jax_influence import memory_utils
from jax_influence.types import ArnoldiEigens
from jax_influence.types import ArnoldiResult
from jax_influence.types import Array
from jax_influence.types import JaxClient
from jax_influence.types import PyTree
from jax_influence.types import Scalar
import numpy as np


def arnoldi_iter(matrix_fn: Callable[[PyTree], PyTree],
                 dot_product_fn: Callable[[PyTree, PyTree], Scalar],
                 start_vector: PyTree,
                 n_iters: int,
                 move_to_host: bool = False,
                 log_progress: bool = False,
                 norm_constant: float = 1.0,
                 stop_tol: float = 1e-6) -> ArnoldiResult:
  """Applies Arnoldi's algorithm.

  Args:
    matrix_fn: Function representing application of a matrix A. Given a PyTree
      v, matrix_fn(v) returns Av.
    dot_product_fn: Function computing the dot product between two pytrees.
    start_vector: PyTree used to start the iteration.
    n_iters: Number of Arnoldi iterations.
    move_to_host: If True offloads the projectors to the CPU as they are
      computed. In this way projections are stored on the host RAM.
    log_progress: If True logs progress after every iteration.
    norm_constant: Constant value for the norm of each projection. In some
      situations (e.g. with a large numbers of parameters) it might be advisable
      to set norm_constant > 1 to avoid dividing projection components by a
      large normalization factor.
    stop_tol: Tolerance used to detect early termination.

  Returns:
    The result of the Arnoldi Iteration, containing the Hessenberg
    matrix A' (n_iters x n_iters-1) approximating A
    on the Krylov subspace K, and the projections onto K. If A is Hermitian
    A' will be tridiagonal (up to numerical errors).
  """

  # Initialization.
  proj = []
  appr_mat = jnp.zeros((n_iters, n_iters - 1))
  v0_norm = jnp.sqrt(dot_product_fn(start_vector, start_vector))
  vec0 = linalg_utils.multiply_by_scalar(norm_constant / v0_norm, start_vector)
  if move_to_host:
    vec0 = memory_utils.tohost_as_numpy(vec0)
  proj.append(vec0)

  for n in range(n_iters - 1):
    if log_progress:
      logging.info('Starting Arnoldi@: %s', n)
    vec = matrix_fn(proj[n])
    if move_to_host:
      vec = memory_utils.tohost_as_numpy(vec)
    for j, proj_vec in enumerate(proj):
      appr_mat = appr_mat.at[j, n].set(
          dot_product_fn(vec, proj_vec) / norm_constant**2)
      vec = linalg_utils.add_scalar_multiple(vec, -appr_mat[j, n], proj_vec)

    new_norm = np.sqrt(dot_product_fn(vec, vec))

    # Early termination if the Krylov subspace is invariant within
    # the tolerance.
    if new_norm < stop_tol:
      appr_mat = appr_mat.at[n + 1, n].set(0)
      if move_to_host:
        vec = jax.tree_map(np.zeros_like, vec)
      else:
        vec = jax.tree_map(jnp.zeros_like, vec)
      proj.append(vec)
      break

    appr_mat = appr_mat.at[n + 1, n].set(new_norm / norm_constant)
    vec = linalg_utils.multiply_by_scalar(1.0 / appr_mat[n + 1, n], vec)

    if move_to_host:
      vec = memory_utils.tohost_as_numpy(vec)
    proj.append(vec)
    if log_progress:
      logging.info('Finished Arnoldi@: %s', n)

  return ArnoldiResult(matrix=appr_mat, proj=proj)


def distill(result: ArnoldiResult,
            top_k: int,
            init_fn: Callable[[Array], Array] = np.zeros_like,
            force_hermitian: bool = True,
            log_progress=False,
            defrag_client: Optional[JaxClient] = None) -> ArnoldiEigens:
  """Distills the result of an Arnoldi iteration to top_k eigenvalues / vectors.

  Args:
    result: Output of the Arnoldi iteration.
    top_k: How many eigenvalues / vectors to distill.
    init_fn: initialization for projection decomposition. With the default
      np.zeros_like the results will live on the host RAM.
    force_hermitian: If True, it assumes that the ArnoldiResult corresponds to a
      Hermitian matrix. Note that if force_hermitian is False, the eigenvalues
      can be complex numbers.
    log_progress: if True logs progress.
    defrag_client: optional JaX Client for memory defragmentation.

  Returns:
    An ArnoldiEigens consisting of the distilled eigenvalues and eigenvectors.
    The eigenvectors are conveniently returned in the original basis used
    before applying the Arnoldi iteration and not wrt. the projectors' basis
    selected for the Krylov subspace.
  """
  appr_mat = result.matrix[:-1, :]
  appr_mat = np.array(appr_mat)
  n = appr_mat.shape[0]

  if force_hermitian:
    # Make appr_mat Hermitian and tridiagonal when force_hermitian=True.
    for i in range(n):
      for j in range(n):
        if i - j > 1 or j - i > 1:
          appr_mat[i, j] = 0
    # Make Hermitian.
    appr_mat = .5 * (appr_mat + appr_mat.T)
    # Get eigenvalues / vectors for Hermitian matrix.
    eigvals, eigvecs = np.linalg.eigh(appr_mat)
  else:
    eigvals, eigvecs = np.linalg.eig(appr_mat)
  # Sort the eigvals by absolute value.
  idx = np.argsort(np.abs(eigvals))
  eigvals = eigvals[idx]
  eigvecs = eigvecs[:, idx]

  # Note we need to discard the last projector as this is a correction term.
  reduced_projections = linalg_utils.change_basis_of_projections(
      eigvecs[:, -top_k:],
      result.proj[:-1],
      init_fn=init_fn,
      defrag_client=defrag_client,
      log_progress=log_progress)

  return ArnoldiEigens(eigenval=eigvals[-top_k:], eigenvec=reduced_projections)

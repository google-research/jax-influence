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

"""Linear Algebra Utilities for PyTrees."""

from typing import Callable, Iterable, List, Optional, Tuple, Union

from absl import logging
import jax
import jax.numpy as jnp
from jax_influence.types import Array
from jax_influence.types import JaxClient
from jax_influence.types import PRNGKey
from jax_influence.types import PyTree
from jax_influence.types import Scalar
from jax_influence.types import Shape
import numpy as np


def multiply_by_scalar(scalar: Scalar, pytree: PyTree) -> PyTree:
  """Multiplies by a scalar.

  Args:
    scalar: The scalar multiplying the pytree.
    pytree: PyTree that gets multiplied.

  Returns:
    scalar * pytree.
  """
  return jax.tree_map(lambda x: x * scalar, pytree)


def add_scalar_multiple(pytree: PyTree, scalar: Scalar,
                        pytree_to_add: PyTree) -> PyTree:
  """Adds a multiple of a PyTree to another one.

  Args:
    pytree: PyTree to which the multiple gets added.
    scalar: The scalar multiplying the vector to be added.
    pytree_to_add: PyTree that gets added.

  Returns:
    pytree + scalar * pytree_to_add.
  """
  map_fn = lambda x, y: x + y * scalar
  return jax.tree_multimap(map_fn, pytree, pytree_to_add)


def inner_product(pytree_a: PyTree,
                  pytree_b: PyTree,
                  use_jax: bool = True) -> Scalar:
  """Computes the inner product of two pytrees.

  Args:
    pytree_a: First PyTree.
    pytree_b: Second PyTree.
    use_jax: If True it uses jax.numpy, otherwise it uses numpy.

  Returns:
    Inner product of pytree_a and pytree_b; concretely computes the
    inner product for each pair of (corresponding) leaf arrays and
    sums all the resulting inner products. The resulting type is a Scalar
    depending on the backend: jnp.ndarray for JaX and a float for numpy backend.
  """
  if use_jax:
    sum_fn = jnp.sum
    add_fn = jnp.add
  else:
    sum_fn = np.sum
    add_fn = np.add
  res = jax.tree_util.tree_multimap(lambda a, b: sum_fn(a * b), pytree_a,
                                    pytree_b)
  return jax.tree_util.tree_reduce(add_fn, res)


def project(proj: Iterable[PyTree],
            pytree: PyTree,
            use_jax: bool = True,
            norm_const: float = 1.0) -> Union[jnp.ndarray, np.ndarray]:
  """Projects pytree along proj.

  Args:
    proj: Iterable of PyTrees giving the projection components.
    pytree: PyTree to be projected.
    use_jax: If True it uses jax.numpy, otherwise it uses numpy.
    norm_const: Optional normalization constant for the projection.

  Returns:
    An Array out where out[i] = inner_product(proj[i], pytree)/norm_const.
    The resulting array type depends on whether jax.numpy / numpy is used.
    Note that unless proj is an orthonormal basis and norm_const=1.0,
    this does NOT correspond to the projection on the subspace spanned by
    the elements of proj.
  """
  out = []
  for p in proj:
    out.append(inner_product(p, pytree, use_jax=use_jax) / norm_const)

  if use_jax:
    return jnp.array(out)
  return np.array(out)


def decompose_by_projection(
    proj: Iterable[PyTree],
    pytree: PyTree,
    use_jax: bool = True,
    norm_const: float = 1.0) -> Tuple[Union[jnp.ndarray, np.ndarray], PyTree]:
  """Computes projection and orthogonal complement.

  Args:
    proj: Iterable of PyTrees giving the projection components.
    pytree: PyTree to be projected.
    use_jax: If True it uses jax.numpy, otherwise it uses numpy.
    norm_const: Optional normalization constant for the projection.

  Returns:
    A tuple (tree_proj, tree_perp) where tree_proj is the projection of pytree
    via proj, and tree_perp is the orthogonal complement. Note that division by
    norm_const happens also when computing the orthogonal complement. This
    gives the correct result if proj consists of orthogonal elements
    each one having norm equal to norm_const.
  """

  tree_proj = project(proj, pytree, norm_const=norm_const, use_jax=use_jax)
  tree_perp = pytree
  for i, p in enumerate(proj):
    tree_perp = add_scalar_multiple(tree_perp, -tree_proj[i] / norm_const, p)

  return tree_proj, tree_perp


def change_basis_of_projections(
    matrix: Array,
    proj: List[PyTree],
    init_fn: Callable[[Array], Array] = np.zeros_like,
    log_progress=False,
    defrag_client: Optional[JaxClient] = None) -> List[PyTree]:
  """Changes basis of projections.

  Given a set of projections `proj` and a transformation matrix `matrix`,
  it allows one to obtain new projections from the composition of `matrix` and
  `proj`. For example, the Arnoldi iteration returns a tridiagonal matrix M and
  a set of projections Q; however, to obtain approximate eigenvectors on the
  Krylov subspace, one needs to diagonalize M and use the change of basis matrix
  to its eigenvectors to combine the projections in Q.
  Args:
    matrix: The matrix to use to compose projections.
    proj: The projections.
    init_fn: A function to init the result. Use np.zeros_like to offload results
      to the host RAM.
    log_progress: If True logs progress.
    defrag_client: Optional JaX client for defragmentation.

  Returns:
    The new projections obtained by composing the old ones with matrix.
  """
  if matrix.shape[0] != len(proj):
    raise ValueError('Incompatible composition')

  out = []
  for j in range(matrix.shape[1]):
    if log_progress:
      logging.info('Compose projections: j=%d', j)
    element = jax.tree_map(init_fn, proj[0])
    if defrag_client is not None:
      defrag_client.defragment()
    for i in range(matrix.shape[0]):
      if log_progress:
        logging.info('Compose projections: i,j=%d,%d', i, j)
      element = add_scalar_multiple(
          scalar=matrix[i, j], pytree=element, pytree_to_add=proj[i])
      if defrag_client is not None:
        defrag_client.defragment()

    out.append(element)
  return out


def random_initialization(
    pytree: PyTree,
    seed: PRNGKey,
    sampler: Callable[[PRNGKey, Shape], Array] = jax.random.normal) -> PyTree:
  """Creates a random pytree with same structure as pytree.

  Args:
    pytree: The model PyTree.
    seed: PRKGKey for initialization.
    sampler: Function used to sample arrays.

  Returns:
    Random pytree with same structure as pytree.
  """
  # Split seeds for all the params
  init_seeds = jax.random.split(seed, num=len(jax.tree_leaves(pytree)))
  init_seeds = jax.tree_unflatten(
      treedef=jax.tree_structure(pytree), leaves=init_seeds)
  random_pytree = jax.tree_multimap(lambda x, y: sampler(y, x.shape), pytree,
                                    init_seeds)

  return random_pytree

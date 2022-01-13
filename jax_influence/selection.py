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

"""Utilities for selecting subsets of parameters."""

from typing import Mapping, Union, Tuple

import flax
from flax import traverse_util
import jax
import jax.numpy as jnp
from jax_influence.types import PyTree
from jax_influence.types import SelectionFn


def split_params(params: PyTree,
                 select_path_fn: SelectionFn) -> Tuple[PyTree, PyTree]:
  """Decomposes parameters in two pieces using a selection function.

  Args:
    params: Frozen dict of parameters.
    select_path_fn: Evaluates to True for the parameter paths to be selected.

  Returns:
    A Tuple (`selected', `unselected'), where the first contains the parameters
    taken by the selection function and the second contains the
    remaining parameters.
  """
  flattened = traverse_util.flatten_dict(flax.core.unfreeze(params))
  selected, unselected = dict(), dict()
  for k, v in flattened.items():
    if select_path_fn(k):
      selected[k] = v
    else:
      unselected[k] = v
  selected = traverse_util.unflatten_dict(selected)
  unselected = traverse_util.unflatten_dict(unselected)
  return flax.core.FrozenDict(selected), flax.core.FrozenDict(unselected)


def merge_params(params_left: PyTree, params_right: PyTree) -> PyTree:
  """Merges two dictionaries of parameters.

  Args:
    params_left: First PyTree.
    params_right: Second PyTree.

  Returns:
    The merge of the two parameter PyTrees.

  """
  out = traverse_util.flatten_dict(flax.core.unfreeze(params_left))
  params_right = traverse_util.flatten_dict(flax.core.unfreeze(params_right))
  for k, v in params_right.items():
    assert k not in params_left
    out[k] = v
  out = traverse_util.unflatten_dict(out)
  return flax.core.FrozenDict(out)


def param_size(params: PyTree) -> int:
  """Computes the total size of parameters."""

  sizes = jax.tree_map(jnp.size, params)
  return sum(jax.tree_leaves(sizes))


def summarize_split_effect(
    params: PyTree,
    select_fn: SelectionFn) -> Mapping[str, Union[str, int, float]]:
  """Summarizes the effect of splitting parameters."""
  total_size = param_size(params)
  sel, _ = split_params(params, select_fn)
  selected_size = param_size(sel)
  out = {}
  out['total_size'] = total_size
  out['pretty_total_size'] = f'{total_size:.3e}'
  out['selected_size'] = selected_size
  out['pretty_selected_size'] = f'{selected_size:.3e}'
  out['selected%'] = selected_size / total_size
  return out

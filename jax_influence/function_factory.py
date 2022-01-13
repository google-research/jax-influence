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

"""A functional factory.

We use functions that take functions and return other functions
to generates things like Hessian Vector Products, Jacobian Vector Products,
etc...
"""
import functools
from typing import Iterator, Optional

import flax
import jax
import jax.numpy as jnp
from jax_influence import batch_utils
from jax_influence import selection
from jax_influence.types import ArrayFn
from jax_influence.types import Batch
from jax_influence.types import Fn
from jax_influence.types import HVPFn
from jax_influence.types import PyTree
from jax_influence.types import ScalarFn
from jax_influence.types import SelectionFn


def restrict_subset_params(fun: Fn, params: PyTree,
                           select_fn: SelectionFn) -> Fn:
  """Restricts a function to be dependent only on a subset of params.

  Args:
    fun: The function to bind.
    params: Full set of parameters.
    select_fn: The function selecting the subset of parameters on which we want
      the resulting function to depend on.

  Returns:
    The function f so that f(x) = f(merge_params(x, unsel)) where unsel are the
    components in params not selected by select_fn.
  """
  _, unsel = selection.split_params(params, select_fn)

  def out_fn(sel, *args, **kwargs):
    new_params = selection.merge_params(sel, unsel)
    return fun(new_params, *args, **kwargs)

  return out_fn


def create_hvp_on_sample(loss_fn: ScalarFn,
                         batch_stats: Optional[PyTree] = None,
                         params_select_fn: Optional[SelectionFn] = None,
                         params_to_bind: Optional[PyTree] = None,
                         mean_across_hosts: bool = False) -> HVPFn:
  """Creates Hessian-vector product on a single sample.

  Args:
    loss_fn: A loss function. If some form of regularization was used during
      training, it is up to the user to include that in loss_fn.
    batch_stats: If loss_fn accepts batch statistics they need to be supplied
      here.
    params_select_fn: A selection function that can be used to select a subset
      of parameters.
    params_to_bind: If params_select_fn is not None, one needs to supply
      parameters to extract those not selected and bind them to be constant
      during differentiation.
    mean_across_hosts: If True it takes the mean across hosts in a multi-host
      environment.

  Returns:
    The Hessian-vector product function with parameters the parameters (in
    JaX terminology called primals), the tangents and the batch.
  """
  raw_loss_fn_ = loss_fn

  def hvp_on_sample(primals, tangents, batch):
    """Computes E_batch [H(primals)(tangents)]."""
    # Bind batch.
    loss_fn_ = functools.partial(raw_loss_fn_, batch=batch)
    # Bind batch_stats if it exists.
    if batch_stats is not None:
      loss_fn_ = functools.partial(loss_fn_, batch_stats=batch_stats)

    if params_select_fn is not None:
      assert params_to_bind is not None, ('For a subset of parameters there '
                                          'must be some to bind.')
      loss_fn_ = restrict_subset_params(loss_fn_, params_to_bind,
                                        params_select_fn)

    grad_fn = jax.grad(loss_fn_)
    loss_hvp = jax.jvp(grad_fn, (primals,), (tangents,))[1]

    if mean_across_hosts:
      loss_hvp = jax.lax.pmean(loss_hvp, 'batch')
    return loss_hvp

  return hvp_on_sample


def create_jvp_on_each_example(
    loss_fn: ArrayFn,
    batch_stats: Optional[PyTree] = None,
    params_select_fn: SelectionFn = None,
    params_to_bind: Optional[PyTree] = None) -> ArrayFn:
  """Creates a Jacobian-vector product on each example.

  Args:
    loss_fn: A loss function WITH per-example loss.
    batch_stats: If loss_fn accepts batch_stats they need to be supplied here.
    params_select_fn: A selection function that can be used to select a subset
      of parameters.
    params_to_bind: If params_select_fn is not None, one needs to supply
      parameters to extract those not selected and bind them to be constant
      during differentiation.

  Returns:
     The function computing the Jacobian-vector product on each example of
       a batch.
  """
  raw_loss_fn_ = loss_fn

  def jvp_on_each_example(primals, tangents, batch):
    """Computes grad(primals).T(tangents)."""
    # Bind batch.
    loss_fn_ = functools.partial(raw_loss_fn_, batch=batch)
    # Bind batch_stas if it exists
    if batch_stats is not None:
      loss_fn_ = functools.partial(loss_fn_, batch_stats=batch_stats)

    if params_select_fn is not None:
      assert params_to_bind is not None, ('For a subset of parameters there '
                                          'must be some to bind.')
      loss_fn_ = restrict_subset_params(loss_fn_, params_to_bind,
                                        params_select_fn)

    loss_jvp = jax.jvp(loss_fn_, (primals,), (tangents,))[1]
    return loss_jvp

  return jvp_on_each_example


def bind_params(fn: Fn, params: PyTree) -> Fn:
  """Binds function to parameters.

  Args:
    fn: Function to bind.
    params: PyTree of parameters to bind. We assume it is the first argument of
      fn.

  Returns:
    The bound version of fn.
  """

  return functools.partial(fn, params)


def create_accumulator(fn: Fn,
                       num_micro_batches: int,
                       do_average: bool = True) -> Fn:
  """Accumulates results across micro batches.

  This is useful if fn cannot be run on the full batch because of memory
  constraints but results can be accumulated across micro batches.

  Args:
    fn: The function to run.
    num_micro_batches: How many micro batches to split each batch.
    do_average: If True averages results on micro batches; otherwise it sums
      them.

  Returns:
    Function performing the desired accumulation.
  """

  if not num_micro_batches > 1:
    raise ValueError('No sense in accumulating with <=1 micro batches')

  def accum_fn(batch, **kwargs):
    batch_size = batch_utils.get_batch_size(batch)
    assert batch_size % num_micro_batches == 0, ('batch size not divisible by '
                                                 'num_micro_batches')
    micro_batch_size = batch_size // num_micro_batches

    def accum_step(loop_cnt: int, accum: PyTree) -> PyTree:
      """Performs an accumulation step."""
      mbatch = batch_utils.get_microbatch(batch, loop_cnt, micro_batch_size)

      result = fn(batch=mbatch, **kwargs)

      accum = jax.tree_multimap(jnp.add, accum, result)
      return accum

    # Initialize accumulator.
    first_batch = batch_utils.get_microbatch(batch, 0, micro_batch_size)
    accum_init = fn(batch=first_batch, **kwargs)
    # Change to float32 for numerical stability.
    accum_dtype = jnp.float32
    accum_init = jax.tree_map(lambda x: x.astype(accum_dtype), accum_init)

    # Run accumulation loop.
    accumulated = jax.lax.fori_loop(1, num_micro_batches, accum_step,
                                    accum_init)
    # Normalize.
    if do_average:
      accumulated = jax.tree_map(lambda x: x / num_micro_batches, accumulated)
    return accumulated

  return accum_fn


def create_hvp_estimator(hvp_fn: HVPFn, handle_p_mapping: bool, params: PyTree,
                         data_iterator: Iterator[Batch]) -> ArrayFn:
  """Creates an HVP estimator consuming an iterator.

  Args:
    hvp_fn: A function performing an HVP step on a batch.
    handle_p_mapping: If True handles replicating parameters and sharding the
      batch across multiple devices.
    params: The parameters at which the HVP gets estimated.
    data_iterator: An Iterator yielding batches.

  Returns:
    The HVP estimator accepting as input the vector on which to evaluate the
    HVP.
  """
  if handle_p_mapping:
    params = flax.jax_utils.replicate(params)

  def compute_fn(vector):
    result = None
    if handle_p_mapping:
      vector = flax.jax_utils.replicate(vector)

    batch = next(data_iterator)
    if handle_p_mapping:
      batch = batch_utils.shard(batch)

    result = hvp_fn(primals=params, tangents=vector, batch=batch)

    if handle_p_mapping:
      result = flax.jax_utils.unreplicate(result)

    return result

  return compute_fn

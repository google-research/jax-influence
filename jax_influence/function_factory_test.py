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

"""Tests for function_factory."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_influence import function_factory
from jax_influence import selection
from jax_influence import test_utils
import numpy as np


def _per_example_loss_fn(params, batch, batch_stats=None):
  """Returns loss on each example."""
  out = batch['x'] @ params['A']
  out = out + batch['y'] @ params['B']
  out = out + params['c']
  # Batch stats is used as a multiplicative constant.
  if batch_stats is not None:
    out = out * batch_stats
  return out**2 / 2


def _total_loss(params, batch, batch_stats=None):
  """Returns loss on a batch."""

  # Batch stats is used as normalization
  if batch_stats is not None:
    den = batch_stats
  else:
    den = 1
  return jnp.sum(_per_example_loss_fn(params, batch)) / den


class FunctionFactoryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.params = {
        'A': jnp.array([1.0, 2.0, -3.0]),
        'B': jnp.array([4.0, 6.0, -3.0, -2.0, -1.0]),
        'c': jnp.array([1.3])
    }
    self.tangents = {
        'A': jnp.array([-0.2, 2.4, 3.22]),
        'B': jnp.array([2.0, 3.0, 1.2, -2.5, -1.44]),
        'c': jnp.array([4.22])
    }

  def get_random_batch(self, seed):
    seed = jax.random.PRNGKey(seed)
    seed1, seed2 = jax.random.split(seed)
    x = jax.random.uniform(seed1, (3,))
    y = jax.random.uniform(seed2, (5,))
    return {'x': x, 'y': y}

  def concat_batches(self, batch0, batch1):
    """Concatenates two batches."""
    return jax.tree_multimap(lambda x, y: jnp.concatenate([x[None], y[None]]),
                             batch0, batch1)

  def get_batch(self):
    """Gets a deterministic batch for testing HVP and JVP."""
    batch0 = {
        'x': jnp.array([1.2, .3, .4]),
        'y': jnp.array([.4, -.5, -2.0, -3.0, -4.2])
    }
    batch1 = {
        'x': jnp.array([3.0, 2.3, 4.4]),
        'y': jnp.array([2.4, -1.5, 2.55, 3.44, -1.0])
    }
    return self.concat_batches(batch0, batch1)

  @parameterized.named_parameters(
      dict(testcase_name='seed_0', seed=0),
      dict(testcase_name='seed_17', seed=17),
  )
  def test_restrict_subset_params(self, seed):
    batch = self.get_random_batch(seed)
    fval = _per_example_loss_fn(self.params, batch)
    fun_rest = function_factory.restrict_subset_params(
        _per_example_loss_fn, self.params, select_fn=lambda x: 'B' in x)
    fvalrest = fun_rest({'B': self.params['B']}, batch)
    test_utils.check_close(fval, fvalrest)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_sel_no_batch_stats',
          select_fn=None,
          batch_stats=None,
          expected={
              'A':
                  np.array([78.24721, 49.799202, 92.40961]),
              'B':
                  np.array(
                      [53.393604, -37.480003, 16.873405, 17.799526, -88.53919]),
              'c':
                  np.array([35.944],)
          }),
      dict(
          testcase_name='no_sel_batch_stats',
          select_fn=None,
          batch_stats=5,
          expected={
              'A':
                  np.array([15.649442, 9.959841, 18.481922]),
              'B':
                  np.array(
                      [10.67872, -7.496001, 3.374681, 3.5599053, -17.707838]),
              'c':
                  np.array([7.1888])
          }),
      dict(
          testcase_name='sel_no_batch_stats',
          select_fn=lambda x: 'B' in x,
          batch_stats=None,
          expected={
              'B':
                  np.array(
                      [-4.9407997, 0.4759996, -30.585999, -44.416, -40.0816])
          }),
      dict(
          testcase_name='sel_batch_stats',
          select_fn=lambda x: 'A' in x,
          batch_stats=3,
          expected={
              'A': np.array([19.795202, 14.810934, 28.231468]),
          }))
  def test_create_hvp_on_sample(self, select_fn, batch_stats, expected):
    hvp_on_sample = function_factory.create_hvp_on_sample(
        _total_loss,
        batch_stats=batch_stats,
        params_select_fn=select_fn,
        params_to_bind=self.params)
    batch = self.get_batch()
    if select_fn is not None:
      sel_params, _ = selection.split_params(self.params, select_fn)
      sel_tangents, _ = selection.split_params(self.tangents, select_fn)
    else:
      sel_params = self.params
      sel_tangents = self.tangents
    result = hvp_on_sample(sel_params, sel_tangents, batch)
    test_utils.check_close(result, expected, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_sel_no_batch_stats',
          select_fn=None,
          batch_stats=None,
          expected=np.array([274.4812, -336.12286])),
      dict(
          testcase_name='no_sel_batch_stats',
          select_fn=None,
          batch_stats=5,
          expected=np.array([6862.0293, -8403.071])),
      dict(
          testcase_name='sel_no_batch_stats',
          select_fn=lambda x: 'B' in x,
          batch_stats=None,
          expected=np.array([174.48161, 65.47399])),
      dict(
          testcase_name='sel_batch_stats',
          select_fn=lambda x: 'A' in x,
          batch_stats=3,
          expected=np.array([265.73044, -2959.9763]),
      ))
  def test_create_jvp_on_each_example(self, select_fn, batch_stats, expected):
    jvp_on_each_example = function_factory.create_jvp_on_each_example(
        _per_example_loss_fn,
        batch_stats=batch_stats,
        params_select_fn=select_fn,
        params_to_bind=self.params)
    batch = self.get_batch()
    if select_fn is not None:
      sel_params, _ = selection.split_params(self.params, select_fn)
      sel_tangents, _ = selection.split_params(self.tangents, select_fn)
    else:
      sel_params = self.params
      sel_tangents = self.tangents
    result = jvp_on_each_example(sel_params, sel_tangents, batch)
    test_utils.check_close(result, expected, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='seed_0', seed=0),
      dict(testcase_name='seed_49', seed=49),
  )
  def test_bind_params(self, seed):
    batch = self.get_random_batch(seed)
    fval = _per_example_loss_fn(self.params, batch)
    fun_bound = function_factory.bind_params(_per_example_loss_fn, self.params)
    fvalbound = fun_bound(batch)
    test_utils.check_close(fval, fvalbound)

  @parameterized.named_parameters(
      dict(testcase_name='no_avg', do_average=False),
      dict(testcase_name='avg', do_average=True),
  )
  def test_create_accumulator(self, do_average):
    batch = self.get_batch()
    expected = _per_example_loss_fn(self.params, batch)
    expected = jnp.sum(expected)
    if do_average:
      expected = expected / 2
    accum_fn = function_factory.create_accumulator(
        _per_example_loss_fn, num_micro_batches=2, do_average=do_average)
    result = accum_fn(params=self.params, batch=batch)
    test_utils.check_close(result, expected)

  @parameterized.named_parameters(
      dict(testcase_name='pmap', handle_p_mapping=True),
      dict(testcase_name='nopmap', handle_p_mapping=False),
  )
  @mock.patch('jax.local_device_count')
  def test_create_hvp_estimator(self, jax_mock, handle_p_mapping):

    # Enforce using only one device when resizing batches.
    jax_mock.return_value = 1

    iter1 = test_utils.InfiniteIterator(0, self.get_random_batch)
    iter2 = test_utils.InfiniteIterator(0, self.get_random_batch)
    hvp_on_sample = function_factory.create_hvp_on_sample(_total_loss)
    batch1 = next(iter1)
    hvp_on_sample_maybe_pmapped = hvp_on_sample
    if handle_p_mapping:
      hvp_on_sample_maybe_pmapped = jax.pmap(hvp_on_sample, 'batch')
    hvp_estimator = function_factory.create_hvp_estimator(
        hvp_on_sample_maybe_pmapped,
        handle_p_mapping=handle_p_mapping,
        params=self.params,
        data_iterator=iter2)
    result = hvp_estimator(self.tangents)
    expected = hvp_on_sample(
        primals=self.params, tangents=self.tangents, batch=batch1)

    test_utils.check_close(result, expected)

if __name__ == '__main__':
  absltest.main()

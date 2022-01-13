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

"""Tests for batch_utils."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_influence import batch_utils
from jax_influence import test_utils
import numpy as np
import tensorflow as tf


class BatchUtilsTest(parameterized.TestCase):

  def get_np_batch(self):
    return {
        'a': np.arange(16 * 22 * 5).reshape((16, 22, 5)),
        'b': np.arange(16 * 39 * 7).reshape((16, 39, 7))
    }

  def get_jnp_batch(self):
    return jax.tree_map(jnp.array, self.get_np_batch())

  def get_tf_batch(self):
    batch = {'a': tf.range(16 * 22 * 5), 'b': tf.range(16 * 39 * 7)}
    batch['a'] = tf.reshape(batch['a'], (16, 22, 5))
    batch['b'] = tf.reshape(batch['b'], (16, 39, 7))
    return batch

  def test_get_first_batch_element_from_dict(self):
    for fn in (self.get_np_batch, self.get_jnp_batch, self.get_tf_batch):
      batch = fn()
      element = batch_utils.get_first_batch_element(batch)
      test_utils.check_equal(element, batch['a'])

  def test_get_first_batch_element_from_array(self):
    for fn in (self.get_np_batch, self.get_jnp_batch):
      batch = fn()
      batch = batch['a']
      element = batch_utils.get_first_batch_element(batch)
      test_utils.check_equal(element, batch)

  def test_get_microbatch(self):
    for fn in (self.get_np_batch, self.get_jnp_batch):
      batch = fn()
      idx = 1
      micro_batch_size = 4
      micro_batch = batch_utils.get_microbatch(batch, idx, micro_batch_size)
      expected = {
          k:
          v[idx * micro_batch_size:(idx * micro_batch_size + micro_batch_size)]
          for k, v in batch.items()
      }
      test_utils.check_equal(expected, micro_batch)

  def test_get_batch_size(self):
    for fn in (self.get_np_batch, self.get_jnp_batch):
      batch = fn()
      expected = 16
      batch_size = batch_utils.get_batch_size(batch)
      self.assertEqual(batch_size, expected)

  def test_maybe_convert_to_array(self):
    batch = self.get_tf_batch()
    batch_jnp = batch_utils.maybe_convert_to_array(batch)
    batch_np = batch_utils.maybe_convert_to_array(batch, np.array)

    batch_jnp_first = batch_utils.get_first_batch_element(batch_jnp)
    self.assertIsInstance(batch_jnp_first, jnp.ndarray)

    batch_np_first = batch_utils.get_first_batch_element(batch_np)
    self.assertIsInstance(batch_np_first, np.ndarray)

    expected = self.get_np_batch()
    test_utils.check_equal(batch_np, expected)
    test_utils.check_equal(batch_jnp, expected)

  @parameterized.named_parameters(
      dict(testcase_name='1device', local_devices=1),
      dict(testcase_name='4devices', local_devices=4),
  )
  @mock.patch('jax.local_device_count')
  def test_shard(self, jax_mock, local_devices):
    jax_mock.return_value = local_devices
    batch = self.get_np_batch()
    sharded = batch_utils.shard(batch)

    # Check leading shape is correct.
    first_shapes = jax.tree_map(lambda x: x.shape[0], sharded)
    test_utils.check_equal(first_shapes,
                           jax.tree_map(lambda x: local_devices, first_shapes))

    # Check trailing shape is correct.
    last_shapes = jax.tree_map(lambda x: x.shape[2:], sharded)
    last_expected = jax.tree_map(lambda x: x.shape[1:], batch)
    for shape1, shape2 in zip(last_shapes.values(), last_expected.values()):
      self.assertTupleEqual(shape1, shape2)

    # Check same number of elements.
    batch_num_elem = jax.tree_map(lambda x: np.prod(x.shape), batch)
    sharded_num_elem = jax.tree_map(lambda x: np.prod(x.shape), sharded)
    test_utils.check_equal(batch_num_elem, sharded_num_elem)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for selection."""

from absl.testing import absltest
import jax.numpy as jnp
from jax_influence import selection
from jax_influence import test_utils


class SelectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params = {'enc': jnp.ones((3, 4)), 'dec': jnp.ones((3, 5, 7))}

  def test_split_params(self):
    sel, unsel = selection.split_params(self.params, lambda x: 'enc' in x)
    test_utils.check_equal(sel, {'enc': jnp.ones((3, 4))})
    test_utils.check_equal(unsel, {'dec': jnp.ones((3, 5, 7))})

  def test_merge_params(self):
    sel, unsel = selection.split_params(self.params, lambda x: 'enc' in x)
    merged = selection.merge_params(sel, unsel)
    test_utils.check_equal(merged, self.params)

  def test_param_size(self):
    size = selection.param_size(self.params)
    self.assertEqual(size, 117)


if __name__ == '__main__':
  absltest.main()

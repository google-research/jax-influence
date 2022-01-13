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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""jax-influence pip package configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

REQUIRED_PACKAGES = ["absl-py", "numpy", "scipy", "jax", "jaxlib", "tensorflow",
                     "flax"]

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="jax-influence",
    version="0.1",
    description="Jax Influence.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/jax-influence",
    author="Google Inc.",
    packages=setuptools.find_packages(),
    license="Apache 2.0",
    install_requires=REQUIRED_PACKAGES,
)

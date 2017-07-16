# -*- coding: utf-8 -*-
#
# Copyright 2017 Daniel Foreman-Mackey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__version__ = "0.0.1.dev0"

try:
    __FINUFFT_SETUP__
except NameError:
    __FINUFFT_SETUP__ = False

if not __FINUFFT_SETUP__:
    __all__ = [
        "nufft1d1", "nufft1d2", "nufft1d3",
        "nufft2d1", "nufft2d2", "nufft2d3",
        "nufft3d1", "nufft3d2", "nufft3d3",
    ]

    from .interface import (
        nufft1d1, nufft1d2, nufft1d3,
        nufft2d1, nufft2d2, nufft2d3,
        nufft3d1, nufft3d2, nufft3d3,
    )

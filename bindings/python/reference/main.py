# Copyright 2017 International Business Machines Corporation
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

import numpy as np
import homotopy
import irls

A = np.array([
    [0.25,  0.25,  0.29,  0.15,  0.14],
    [0.20,  0.15,  0.02,  0.16,  0.27],
    [0.15,  0.16,  0.29,  0.07,  0.09],
    [0.12,  0.25,  0.07,  0.25,  0.28],
    [0.20,  0.17,  0.29,  0.25,  0.14]],
    dtype=np.float32
)

b = np.asarray(
    [0.27,  0.12,  0.25,  0.02,  0.27],
    dtype=np.float32
)

tolerance = 0.05


print("--\n-- Reference implementations\n--\n")
print("-- Homotopy\n")

x = homotopy.solve(A, b, 5, tolerance)
print("x={}\nargmax(x)={}".format(x, np.argmax(x)))

print("\n-- IRLS\n")

x = irls.solve(A, b, 5, tolerance)
print("x={}\nargmax(x)={}".format(x, np.argmax(x)))
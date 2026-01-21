/*
 * Copyright 2026 Nicolas Spijkerman
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::math::Matrix;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Layer {
    weights: Matrix,
    biases: Matrix,
}

impl Layer {
    pub fn new<F: FnMut() -> f32>(input_size: usize, output_size: usize, mut rng: F) -> Self {
        Self {
            weights: Matrix::new_random(output_size, input_size, &mut rng),
            biases: Matrix::new_random(output_size, 1, &mut rng),
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        &self.weights * input + &self.biases
    }
}

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

use serde::{Deserialize, Serialize};
use std::{
    iter,
    ops::{Add, Mul},
};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new<T: Into<Vec<f32>>>(rows: usize, cols: usize, data: T) -> Self {
        let data: Vec<f32> = data.into();
        Self { rows, cols, data }
    }

    pub fn new_zero(rows: usize, cols: usize) -> Self {
        let data: Vec<f32> = vec![0.0; rows * cols];
        Self { rows, cols, data }
    }

    pub fn new_random<F: FnMut() -> f32>(rows: usize, cols: usize, rng: F) -> Self {
        let data: Vec<f32> = iter::repeat_with(rng).take(rows * cols).collect();
        Self { rows, cols, data }
    }
}

impl Add<&Self> for Matrix {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        debug_assert_eq!(self.data.len(), rhs.data.len(), "TODO: good error message");

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Self::new(self.rows, self.cols, result)
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        const BLOCK_SIZE: usize = 64;

        debug_assert_eq!(
            self.cols, rhs.rows,
            "matrix dimension mismatch: {}x{} * {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );

        let mut result = Matrix::new_zero(self.rows, rhs.cols);

        for block_row_start in (0..self.rows).step_by(BLOCK_SIZE) {
            for block_col_start in (0..rhs.cols).step_by(BLOCK_SIZE) {
                for block_inner_start in (0..self.cols).step_by(BLOCK_SIZE) {
                    let block_row_end: usize = (block_row_start + BLOCK_SIZE).min(self.rows);
                    let block_col_end: usize = (block_col_start + BLOCK_SIZE).min(rhs.cols);
                    let block_inner_end: usize = (block_inner_start + BLOCK_SIZE).min(self.cols);

                    for row_index in block_row_start..block_row_end {
                        for inner_index in block_inner_start..block_inner_end {
                            let self_value: f32 = self.data[row_index * self.cols + inner_index];
                            let rhs_row_offset: usize = inner_index * rhs.cols;
                            let result_row_offset: usize = row_index * rhs.cols;

                            for col_index in block_col_start..block_col_end {
                                result.data[result_row_offset + col_index] +=
                                    self_value * rhs.data[rhs_row_offset + col_index];
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_matrix_new_zero() {
        let matrix = Matrix::new_zero(2, 2);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![0.0; 4]);
    }

    #[test]
    fn test_matrix_new_random() {
        let matrix = Matrix::new_random(2, 2, || 1.0);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0; 4]);
    }

    #[test]
    #[should_panic(expected = "matrix dimension mismatch")]
    fn test_matrix_mul_dimension_mismatch() {
        let matrix1 = Matrix::new_random(1, 2, || 0.0);
        let matrix2 = Matrix::new_random(1, 2, || 0.0);
        let _ = &matrix1 * &matrix2;
    }

    #[test]
    fn test_matrix_mul() {
        let matrix1 = Matrix::new_random(2, 1, || 2.0);
        let matrix2 = Matrix::new_random(1, 2, || 3.0);
        let result = &matrix1 * &matrix2;
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![6.0; 4]);
    }
}

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

use super::Layer;
use crate::{
    core::{Error, Result},
    math::Matrix,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct MultilayerSingleton {
    layers: Vec<Layer>,
}

impl MultilayerSingleton {
    pub fn new<T: Into<Vec<usize>>, F: FnMut() -> f32>(layer_sizes: T, mut rng: F) -> Self {
        let layer_sizes: Vec<usize> = layer_sizes.into();

        debug_assert!(
            layer_sizes.len() > 1,
            "layer_sizes must contain at least 2 elements to define input and output layers, got {}",
            layer_sizes.len()
        );

        Self {
            layers: layer_sizes
                .windows(2)
                .map(|sizes| Layer::new(sizes[0], sizes[1], &mut rng))
                .collect(),
        }
    }

    pub fn load_from_file<T: AsRef<Path>>(path: T) -> Result<Self> {
        let file = File::open(path).map_err(Error::Io)?;
        let mut reader = BufReader::new(file);

        let mut content = Vec::new();
        Read::read_to_end(&mut reader, &mut content).map_err(Error::Io)?;

        postcard::from_bytes(&content).map_err(Error::Postcard)
    }

    pub fn save_to_file<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        let file = File::create(path).map_err(Error::Io)?;
        let writer = BufWriter::new(file);

        let bytes = postcard::to_stdvec(self).map_err(Error::Postcard)?;

        let mut writer = writer;
        writer.write_all(&bytes).map_err(Error::Io)?;

        Ok(())
    }

    pub fn forward<T: Into<Vec<f32>>>(&self, input: T) -> Vec<f32> {
        let input: Vec<f32> = input.into();
        let mut input: Matrix = Matrix::new(input.len(), 1, input);

        for layer in &self.layers {
            input = layer.forward(&input);
        }

        input.data
    }
}

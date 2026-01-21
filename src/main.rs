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

//! A pure Rust implementation of a multilayer perceptron for MNIST handwritten digit recognition,
//! built entirely from scratch without the use of machine learning frameworks.

mod cli;
mod core;
mod gui;
mod math;
mod neural_network;

use core::Result;
use neural_network::MultilayerSingleton;
use rand::Rng;

fn main() -> Result<()> {
    let args = cli::Args::parse();

    match args {
        cli::Args::Train { file } => {
            let mut rng = rand::rng();
            let neural_network = MultilayerSingleton::new([2, 3, 1], || rng.random());
            neural_network.save_to_file(file)?;
            Ok(())
        }
        cli::Args::App { file } => {
            let mut rng = rand::rng();
            let neural_network = MultilayerSingleton::new([2, 3, 1], || rng.random());
            let app = gui::App::new(neural_network);
            app.run()
        }
    }
}

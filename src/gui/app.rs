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

use crate::{
    core::{Error, Result},
    MultilayerSingleton,
};
use eframe::{
    egui::{Context, ViewportBuilder}, Frame,
    NativeOptions,
};

#[derive(Debug, Default)]
pub struct App {
    neural_network: MultilayerSingleton,
}

impl App {
    pub const fn new(neural_network: MultilayerSingleton) -> Self {
        Self { neural_network }
    }

    pub fn run(self) -> Result<()> {
        eframe::run_native(
            "Handwritten Digit Recognition",
            NativeOptions {
                viewport: ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
                centered: true,
                ..NativeOptions::default()
            },
            Box::new(|_| Ok(Box::new(self))),
        )
        .map_err(Error::Eframe)
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {}
}

use std::env::args;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::wgpu::{GraphicsApi, self};
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi, Fusion};
use burn::tensor::f16;
use models::{GeneratorConfig, DiscriminatorConfig};

mod image;
mod data_loader;
mod models;
mod training;
mod leaky_relu;
mod cross_entropy_loss;

fn main() {
    let args: Vec<String> = args().collect();
    if args.len() > 1 && args[1] == "--bake" {
        data_loader::bake_image_dataset("../../ML_data/img_align_celeba/");
        println!("Baking of Images into Sqlite finished.");
    }
    else {
        run();
    }
}

fn run() {
    type MyBackend = Wgpu<burn::backend::wgpu::OpenGl, f32, i32>;
    // type MyBackend = Wgpu<burn::backend::wgpu::AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Fusion<MyBackend>;
    type FusionAutodiff = Autodiff<MyAutodiffBackend>;

    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
    // type FusionAutodiff = Autodiff<LibTorch<f32>>;
    // let device = LibTorchDevice::Cpu;

    training::train::<FusionAutodiff>(
        "./artifacts",
        training::TrainingConfig::new(GeneratorConfig::new(), DiscriminatorConfig::new()),
        device,
    );

}
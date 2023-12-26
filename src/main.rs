use std::env::args;
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::wgpu::{GraphicsApi, self};
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi, Fusion};
use models::{GeneratorConfig, DiscriminatorConfig};

mod image;
mod data_loader;
mod models;
mod training;

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
    type MyBackend = Wgpu<burn::backend::wgpu::Vulkan, f32, i32>;
    // type MyBackend = Wgpu<burn::backend::wgpu::AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = MyBackend;//Fusion<MyBackend>;
    type FusionAutodiff = Autodiff<MyAutodiffBackend>;

    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
    // type FusionAutodiff = Autodiff<LibTorch<f32>>;
    // let device = LibTorchDevice::Vulkan;

    training::train::<FusionAutodiff>(
        "./artifacts",
        training::TrainingConfig::new(GeneratorConfig::new(), DiscriminatorConfig::new(), AdamConfig::new().with_beta_1(0.5)),
        device,
    );

}
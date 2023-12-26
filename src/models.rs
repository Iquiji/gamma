use burn::{module::Module, config::Config, nn::{conv::{ConvTranspose2d, Conv2d, ConvTranspose2dConfig, Conv2dConfig}, BatchNorm, ReLU, BatchNormConfig, loss::{CrossEntropyLoss, BinaryCrossEntropyLossConfig}, Linear, LinearConfig, Initializer}, tensor::{Tensor, backend::Backend, activation::sigmoid, Int}, train::ClassificationOutput, backend::autodiff::ops::Init};
use crate::leaky_relu::leaky_relu;

// See: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#[derive(Module, Debug)]
pub struct Generator<B: Backend>{
    projection: Linear<B>,
    conv1: ConvTranspose2d<B>,
    batch_norm1: BatchNorm<B,2>,
    conv2: ConvTranspose2d<B>,
    batch_norm2: BatchNorm<B,2>,
    conv3: ConvTranspose2d<B>,
    batch_norm3: BatchNorm<B,2>,
    conv4: ConvTranspose2d<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    #[config(default = "100")]
    pub latent_vector_size: usize,
    #[config(default = "64")]
    feature_map_size: usize,
}

impl GeneratorConfig{
    pub fn init<B: Backend>(&self, conv_initializer: &Initializer) -> Generator<B> {
        Generator { 
            projection: LinearConfig::new(100, 1024 * 4 * 4).init(),
            conv1: ConvTranspose2dConfig::new([self.feature_map_size * 16, self.feature_map_size * 8], [4,4]).with_stride([2, 2]).with_padding([1,1]).with_initializer(conv_initializer.clone()).init(),
            batch_norm1: BatchNormConfig::new(self.feature_map_size * 8).init(),
            conv2: ConvTranspose2dConfig::new([self.feature_map_size * 8, self.feature_map_size * 4], [4,4]).with_stride([2,2]).with_padding([1,1]).with_initializer(conv_initializer.clone()).init(),
            batch_norm2: BatchNormConfig::new(self.feature_map_size * 4).init(),
            conv3: ConvTranspose2dConfig::new([self.feature_map_size * 4, self.feature_map_size * 2], [4,4]).with_stride([2,2]).with_padding([1,1]).with_initializer(conv_initializer.clone()).init(),
            batch_norm3: BatchNormConfig::new(self.feature_map_size * 2).init(),
            conv4: ConvTranspose2dConfig::new([self.feature_map_size * 2, 3], [4,4]).with_stride([2,2]).with_padding([1,1]).with_initializer(conv_initializer.clone()).init(),
        }
    }
}

impl<B: Backend> Generator<B> {
    pub fn forward(&self, latents: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch_size, latents_size] = latents.dims();

        // Create a channel at the second dimension.
        let x = latents;

        let x = self.projection.forward(x);
        let x = x.reshape([batch_size, 1024, 4, 4]);

        // Round 1
        let x = self.conv1.forward(x);
        let x = self.batch_norm1.forward(x);
        let x = leaky_relu(x);
        // Round 2
        let x = self.conv2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = leaky_relu(x);
        // Round 3
        let x = self.conv3.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = leaky_relu(x);
        // Round 4
        let x = self.conv4.forward(x);
        x.tanh() // [batch, 3, height, width]
    }
    pub fn forward_print_sizes(&self, latents: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch_size, latents_size] = latents.dims();
        println!("latents: {:?}", latents.shape());

        // Create a channel at the second dimension.
        let x = latents;

        let x = self.projection.forward(x);
        let x = x.reshape([batch_size, 1024, 4, 4]);
        println!("projection: {:?}", x.shape());

        // Round 1
        let x = self.conv1.forward(x);
        let x = self.batch_norm1.forward(x);
        let x = leaky_relu(x);
        println!("conv1: {:?}", x.shape());
        // Round 2
        let x = self.conv2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = leaky_relu(x);    
        println!("conv2: {:?}", x.shape());
        // Round 3
        let x = self.conv3.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = leaky_relu(x);
        println!("conv3: {:?}", x.shape());
        // Round 4
        let x = self.conv4.forward(x);
        println!("conv4: {:?}", x.shape());
        x.tanh() // [batch, 3, height, width]
    }
}


#[derive(Module, Debug)]
pub struct Discriminator<B: Backend>{
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    norm2: BatchNorm<B,2>,
    conv3: Conv2d<B>,
    norm3: BatchNorm<B,2>,
    conv4: Conv2d<B>,
    norm4: BatchNorm<B,2>,
    final_conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig{
    #[config(default = "64")]
    feature_map_size: usize,
}

impl DiscriminatorConfig{
    pub fn init<B: Backend>(&self, conv_initializer: &Initializer) -> Discriminator<B> {
        Discriminator { 
            conv1: Conv2dConfig::new([3, self.feature_map_size], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).with_initializer(conv_initializer.clone()).init(),
            conv2: Conv2dConfig::new([self.feature_map_size, self.feature_map_size * 2], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).with_initializer(conv_initializer.clone()).init(),
            norm2: BatchNormConfig::new(self.feature_map_size * 2).init(),
            conv3: Conv2dConfig::new([self.feature_map_size * 2, self.feature_map_size * 4], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).with_initializer(conv_initializer.clone()).init(),
            norm3: BatchNormConfig::new(self.feature_map_size * 4).init(),
            conv4: Conv2dConfig::new([self.feature_map_size * 4, self.feature_map_size * 8], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).with_initializer(conv_initializer.clone()).init(),
            norm4: BatchNormConfig::new(self.feature_map_size * 8).init(),
            final_conv: Conv2dConfig::new([self.feature_map_size * 8, 1], [4,4]).with_initializer(conv_initializer.clone()).init() 
        }
    }
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch_size, channels, width, height] = images.dims();

        // Round 1
        let x = self.conv1.forward(images);
        let x = leaky_relu(x);


        // Round 2
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        let x = leaky_relu(x);

        // Round 3
        let x = self.conv3.forward(x); 
        let x = self.norm3.forward(x);
        let x = leaky_relu(x);

        // Round 4
        let x = self.conv4.forward(x);
        let x = self.norm4.forward(x);
        let x = leaky_relu(x);
        
        // Final Conv and Tanh
        let x = self.final_conv.forward(x);
        let x = x.reshape([batch_size]);
        sigmoid(x)
    }
}
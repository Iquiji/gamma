use burn::{module::Module, config::Config, nn::{conv::{ConvTranspose2d, Conv2d, ConvTranspose2dConfig, Conv2dConfig}, BatchNorm, ReLU, BatchNormConfig, loss::{CrossEntropyLoss, BinaryCrossEntropyLossConfig}, Linear, LinearConfig}, tensor::{Tensor, backend::Backend, activation::sigmoid, Int}, train::ClassificationOutput};

// See: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#[derive(Module, Debug)]
pub struct Generator<B: Backend>{
    projection: Linear<B>,
    conv1: ConvTranspose2d<B>,
    batch_norm1: BatchNorm<B,2>,
    relu1: ReLU,
    conv2: ConvTranspose2d<B>,
    relu2: ReLU,
    batch_norm2: BatchNorm<B,2>,
    conv3: ConvTranspose2d<B>,
    batch_norm3: BatchNorm<B,2>,
    relu3: ReLU,
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
    pub fn init<B: Backend>(&self) -> Generator<B> {
        Generator { 
            projection: LinearConfig::new(100, 1024 * 4 * 4).init(),
            conv1: ConvTranspose2dConfig::new([self.feature_map_size * 16, self.feature_map_size * 8], [4,4]).with_stride([2, 2]).with_padding([1,1]).init(),
            batch_norm1: BatchNormConfig::new(self.feature_map_size * 8).init(),
            relu1: ReLU::new(),
            conv2: ConvTranspose2dConfig::new([self.feature_map_size * 8, self.feature_map_size * 4], [4,4]).with_stride([2,2]).with_padding([1,1]).init(),
            batch_norm2: BatchNormConfig::new(self.feature_map_size * 4).init(),
            relu2: ReLU::new(),
            conv3: ConvTranspose2dConfig::new([self.feature_map_size * 4, self.feature_map_size * 2], [4,4]).with_stride([2,2]).with_padding([1,1]).init(),
            batch_norm3: BatchNormConfig::new(self.feature_map_size * 2).init(),
            relu3: ReLU::new(),
            conv4: ConvTranspose2dConfig::new([self.feature_map_size * 2, 3], [4,4]).with_stride([2,2]).with_padding([1,1]).init(),
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
        let x = self.relu1.forward(x);
        // Round 2
        let x = self.conv2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = self.relu2.forward(x);
        // Round 3
        let x = self.conv3.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = self.relu3.forward(x);
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
        let x = self.relu1.forward(x);
        println!("conv1: {:?}", x.shape());
        // Round 2
        let x = self.conv2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = self.relu2.forward(x);    
        println!("conv2: {:?}", x.shape());
        // Round 3
        let x = self.conv3.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = self.relu3.forward(x);
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
    relu1: ReLU,
    conv2: Conv2d<B>,
    norm2: BatchNorm<B,2>,
    relu2: ReLU,
    conv3: Conv2d<B>,
    norm3: BatchNorm<B,2>,
    relu3: ReLU,
    conv4: Conv2d<B>,
    norm4: BatchNorm<B,2>,
    relu4: ReLU,
    final_conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig{
    #[config(default = "64")]
    feature_map_size: usize,
}

impl DiscriminatorConfig{
    pub fn init<B: Backend>(&self) -> Discriminator<B> {
        Discriminator { 
            conv1: Conv2dConfig::new([3, self.feature_map_size], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).init(),
            relu1: ReLU::new(),
            conv2: Conv2dConfig::new([self.feature_map_size, self.feature_map_size * 2], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).init(),
            norm2: BatchNormConfig::new(self.feature_map_size * 2).init(),
            relu2: ReLU::new(),
            conv3: Conv2dConfig::new([self.feature_map_size * 2, self.feature_map_size * 4], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).init(),
            norm3: BatchNormConfig::new(self.feature_map_size * 4).init(),
            relu3: ReLU::new(),
            conv4: Conv2dConfig::new([self.feature_map_size * 4, self.feature_map_size * 8], [4,4]).with_stride([2,2]).with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1)).init(),
            norm4: BatchNormConfig::new(self.feature_map_size * 8).init(),
            relu4: ReLU::new(),
            final_conv: Conv2dConfig::new([self.feature_map_size * 8, 1], [4,4]).init() 
        }
    }
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch_size, channels, width, height] = images.dims();

        // Round 1
        let x = self.conv1.forward(images);
        let x = self.relu1.forward(x);


        // Round 2
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        let x = self.relu2.forward(x);

        // Round 3
        let x = self.conv3.forward(x); 
        let x = self.norm3.forward(x);
        let x = self.relu3.forward(x);

        // Round 4
        let x = self.conv4.forward(x);
        let x = self.norm4.forward(x);
        let x = self.relu4.forward(x);
        
        // Final Conv and Tanh
        let x = self.final_conv.forward(x);
        let x = x.reshape([batch_size]);
        sigmoid(x)
    }
}
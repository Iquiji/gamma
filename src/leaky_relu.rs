use burn::tensor::{backend::Backend, Tensor};

pub fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_max(0) +  x.clamp_min(0).mul_scalar(-0.2)
}
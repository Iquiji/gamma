use burn::tensor::{backend::Backend, Tensor, Float};

pub fn cross_entropy_loss<B: Backend, const D: usize>(targets_float: Tensor<B, D, Float>, logits: Tensor<B, D, Float>) -> Tensor<B, 1, Float> {
    let loss = targets_float.clone() * logits.clone().log()
            + (targets_float.clone().neg() + 1.) * (logits.neg() + 1.).log();
    loss.mean().neg()
}
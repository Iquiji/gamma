use std::{fs, ffi::OsStr};

use burn::{data::dataset::{SqliteDatasetWriter, SqliteDataset}, tensor::DataSerialize};
use image::{GenericImageView, Rgb};
use image::io::Reader as ImageReader;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::{Backend, AutodiffBackend}, Data, ElementConversion, Int, Tensor}, train::{TrainStep, ClassificationOutput, TrainOutput, ValidStep},
};

use crate::image::{IMAGE_HEIGHT, IMAGE_WIDTH};

pub struct ImageBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    // A "List" of 3d Images
    pub images: Tensor<B, 4>,
}

impl<B: Backend> Batcher<DataSerialize<u8>, ImageBatch<B>> for ImageBatcher<B> {
    fn batch(&self, items: Vec<DataSerialize<u8>>) -> ImageBatch<B> {
        let images = items
            .iter()
            .map(|data| data.into())
            .map(|data: Data<u8, 3>| data.clone().convert())
            .map(|data: Data<f32, 3>| Tensor::<B, 3>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 3, IMAGE_WIDTH, IMAGE_HEIGHT]))
            .map(|tensor| (tensor / 255) - 0.5)
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);

        ImageBatch { images }
    }
}



pub fn make_image_dataset() -> SqliteDataset<DataSerialize<u8>>{
    SqliteDataset::from_db_file("training_data.sqlite", "train").unwrap()
}


pub fn bake_image_dataset(path: &str) {
    let mut writer: SqliteDatasetWriter<DataSerialize<u8>> = SqliteDatasetWriter::new("training_data.sqlite", false).unwrap();
    let paths = fs::read_dir(path).unwrap();
    #[allow(clippy::never_loop)]
    for path in paths {
        let path = path.unwrap();
        if OsStr::to_str(path.path().extension().unwrap()).unwrap() == "jpg" {
            let img = ImageReader::open(path.path()).unwrap().decode().unwrap();    
            let img = img.resize(IMAGE_HEIGHT as u32, IMAGE_WIDTH as u32, image::imageops::FilterType::Lanczos3);        
            let mut img_buf = [[[0 ; 3 ]; IMAGE_HEIGHT] ; IMAGE_WIDTH];

            for (x, y, pixel) in img.pixels() {
                img_buf[y as usize][x as usize][0] = pixel.0[0];
                img_buf[y as usize][x as usize][1] = pixel.0[1];
                img_buf[y as usize][x as usize][2] = pixel.0[2];
            }

            let data: Data<u8, 3> = img_buf.into();
            let data = data.serialize();

            // Insert into sqlite
            writer.write("train", &data).unwrap();
            println!("Wrote: {:?}", path.path());
        }
    }
    writer.set_completed().unwrap();
}
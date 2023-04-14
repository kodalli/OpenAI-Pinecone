use std::array::TryFromSliceError;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Debug;
use async_trait::async_trait;

#[async_trait]
pub trait Database: Debug {
    async fn create(&self, id: &str, data: &str) -> Result<(), Box<dyn Error>>;
    async fn read(&self, id: &str) -> Result<String, Box<dyn Error>>;
    async fn update(&self, id: &str, data: &str) -> Result<(), Box<dyn Error>>;
    async fn delete(&self, id: &str) -> Result<(), Box<dyn Error>>;
}

pub enum DatabaseOperation {
    Create,
    Read,
    Update,
    Delete,
}

pub fn convert_binary_to_embeddings(binary_data: &[u8]) -> Result<Vec<f32>, TryFromSliceError> {
    let mut embeddings = Vec::with_capacity(binary_data.len() / 4);
    for chunk in binary_data.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into()?);
        embeddings.push(value);
    }
    Ok(embeddings)
}

pub fn convert_embeddings_to_binary(embeddings: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in embeddings {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

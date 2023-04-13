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

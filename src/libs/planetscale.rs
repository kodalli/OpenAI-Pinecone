use std::collections::HashMap;
use std::error::Error;
use mysql_async::{
    Pool,
    Row,
    prelude::Queryable,
    params,
};
use std::sync::Arc;
use async_trait::async_trait;
use crate::libs::database::{convert_binary_to_embeddings, convert_embeddings_to_binary, Database};

#[derive(Debug)]
pub struct PlanetScaleDB {
    pool: Arc<Pool>,
}

impl PlanetScaleDB {
    pub async fn new(connection_string: &str) -> Result<Self, Box<dyn Error>> {
        let pool = Pool::new(connection_string);
        let observer = PlanetScaleDB { pool: Arc::new(pool) };
        observer.init().await?;
        Ok(observer)
    }

    async fn execute_query(&self, query: &str) -> Result<(), Box<dyn Error>> {
        let mut conn = self.pool.get_conn().await?;
        conn.query_drop(query).await?;
        Ok(())
    }

    async fn init(&self) -> Result<(), Box<dyn Error>> {
        let create_table_query = r#"
            CREATE TABLE IF NOT EXISTS items (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#;
        self.execute_query(create_table_query).await
    }

    async fn insert_embedding_data(&self, id: &str, data: &str, embeddings: &[f32]) {
        let mut conn = self.pool.get_conn().await.unwrap();
        let binary_embeddings = convert_embeddings_to_binary(embeddings);

        let query = r"INSERT INTO text_embeddings (id, data, embedding) VALUES (:id, :data, :embedding)";
        let params = params! {
            "id" => id,
            "data" => data,
            "embedding" => &binary_embeddings,
        };

        conn.exec_drop(query, params).await.unwrap();
    }

    async fn get_embedding_data(&self, id: &str) -> Option<(String, String, Vec<f32>)> {
        let mut conn = self.pool.get_conn().await.unwrap();

        let query = r"SELECT id, data, embedding FROM text_embeddings WHERE id = :id";
        let params = params! {
            "id" => id,
        };

        let row : Option<(String, String, Vec<u8>)> = conn.exec_first(query, params).await.unwrap();
        if let Some((id, data, binary_data)) = row {
            let id: String = id;
            let data: String = data;
            let embeddings = convert_binary_to_embeddings(binary_data.as_slice()).unwrap();
            Some((id, data, embeddings))
        } else {
            None
        }
    }
}

#[async_trait]
impl Database for PlanetScaleDB {
    async fn create(&self, id: &str, data: &str) -> Result<(), Box<dyn Error>> {
        let query = format!("INSERT INTO data_table (id, data) VALUES ('{}', '{}')", id, data);
        self.execute_query(&query).await
    }

    async fn read(&self, id: &str) -> Result<String, Box<dyn Error>> {
        let mut conn = self.pool.get_conn().await?;
        let query = format!("SELECT data FROM data_table WHERE id = '{}'", id);
        let row: Option<Row> = conn.query_first(query).await?;

        match row {
            Some(row) => {
                let data: String = row.get("data").unwrap();
                Ok(data)
            }
            None => Err(Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Data not found"))),
        }
    }

    async fn update(&self, id: &str, data: &str) -> Result<(), Box<dyn Error>> {
        let query = format!("UPDATE data_table SET data = '{}' WHERE id = '{}'", data, id);
        self.execute_query(&query).await
    }

    async fn delete(&self, id: &str) -> Result<(), Box<dyn Error>> {
        let query = format!("DELETE FROM data_table WHERE id = '{}'", id);
        self.execute_query(&query).await
    }
}

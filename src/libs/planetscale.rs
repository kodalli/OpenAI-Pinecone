use std::error::Error;
use mysql_async::{
    Pool,
    Row,
    prelude::Queryable
};
use std::sync::Arc;
use crate::libs::database::Database;
use async_trait::async_trait;

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

    async fn init(&self) -> Result <(), Box<dyn Error>> {
        let create_table_query = r#"
            CREATE TABLE IF NOT EXISTS items (
                id VARCHAR(255) PRIMARY KEY,
                data TEXT NOT NULL,
                embedding BLOB
            )
        "#;
        self.execute_query(create_table_query).await
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

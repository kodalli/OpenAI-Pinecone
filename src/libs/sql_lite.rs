use crate::libs::database::Database;
use rusqlite::{params, Connection};
use std::error::Error;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct SQLiteDB {
    // SQLite database connection details here
    conn: Arc<Mutex<Connection>>,
}

impl SQLiteDB {
    pub fn new(db_name: &str) -> Result<Self, Box<dyn Error>> {
        let conn = Connection::open(db_name)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl Database for SQLiteDB {
    async fn create(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO items (id, data) VALUES (?1, ?2)",
            params![id, data],
        )?;
        Ok(())
    }

    async fn read(&self, id: &str) -> Result<String, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT data FROM items WHERE id = ?1")?;
        let data: String = stmt.query_row(params![id], |row| row.get(0))?;
        Ok(data)
    }

    async fn update(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE items SET data = ?2 WHERE id = ?1",
            params![id, data],
        )?;
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM items WHERE id = ?1",
            params![id]
        )?;
        Ok(())
    }
}

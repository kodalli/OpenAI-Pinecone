use std::{error::Error, fmt::Debug};

use rusqlite::{params, Connection};

#[derive(Debug)]
pub struct SQLiteObserver {
    // SQLite database connection details here
    conn: Connection,
}

impl SQLiteObserver {
    pub fn new(db_name: &str) -> Result<Self, Box<dyn Error>> {
        let conn = Connection::open(db_name)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    // Implement SQLite-related functionality here
}

pub trait Observer: Debug {
    fn create(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn read(&self, id: &str) -> Result<String, Box<dyn std::error::Error>>;
    fn update(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>>;
}

impl Observer for SQLiteObserver {
    fn create(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.execute(
            "INSERT INTO items (id, data) VALUES (?1, ?2)",
            params![id, data],
        )?;
        Ok(())
    }

    fn read(&self, id: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut stmt = self.conn.prepare("SELECT data FROM items WHERE id = ?1")?;
        let data: String = stmt.query_row(params![id], |row| row.get(0))?;
        Ok(data)
    }

    fn update(&self, id: &str, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.execute(
            "UPDATE items SET data = ?2 WHERE id = ?1",
            params![id, data],
        )?;
        Ok(())
    }

    fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.conn
            .execute("DELETE FROM items WHERE id = ?1", params![id])?;
        Ok(())
    }
}

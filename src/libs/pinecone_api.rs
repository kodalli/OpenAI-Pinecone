use lazy_static::lazy_static;
use reqwest::Client;
use serde::de::DeserializeOwned;
use std::{env, sync::Arc};
use reqwest::header::{HeaderMap, HeaderValue};
use thiserror::Error;
use crate::libs::database::DatabaseOperation;

use super::pinecone_data::{IdList, PineconeRequest, PineconeResponse};

lazy_static! {
    static ref CLIENT: Arc<Client> = {
        dotenv::dotenv().ok();
        let api_key = env::var("PINECONE_API_KEY").expect("Failed to locate api key.");

        let client = Client::builder()
            .default_headers(headers(api_key))
            .build()
            .expect("Failed to create client connection.");

        Arc::new(client)
    };
}

fn headers(api_key: String) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("Api-Key", HeaderValue::from_str(api_key.as_str()).unwrap());
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_str("application/json").unwrap(),
    );
    headers.insert(reqwest::header::ACCEPT, HeaderValue::from_str("application/json").unwrap());

    headers
}

const BASE_URL: &str = "https://test-index-1a567db.svc.us-west4-gcp.pinecone.io/";
const UPSERT: &str = "vectors/upsert";
const QUERY: &str = "query";
const UPDATE: &str = "vectors/update";
const FETCH: &str = "vectors/fetch";
const DELETE: &str = "vectors/delete";

// Error handling
#[derive(Debug, Error)]
pub enum PineconeApiError {
    #[error("UpsertError: {0}")]
    UpsertError(String),

    #[error("QueryError: {0}")]
    QueryError(String),

    #[error("UpdateError: {0}")]
    UpdateError(String),

    #[error("FetchError: {0}")]
    FetchError(String),

    #[error("DeleteError: {0}")]
    DeleteError(String),
}
// Error handling

// Request Functions
impl PineconeRequest {
    async fn send<T, E>(&self, endpoint: &str, error: E) -> Result<T, PineconeApiError>
        where
            T: DeserializeOwned,
            E: Fn(String) -> PineconeApiError,
    {
        let response = CLIENT
            .post(format!("{}{}", BASE_URL, endpoint))
            .json(self)
            .send()
            .await;

        println!("{:?}", response);

        let result = match response {
            Ok(response) => {
                let status = response.status();

                if status.is_success() {
                    println!("success");
                    response.json().await.map_err(|e| e.to_string())
                } else {
                    println!("{}", status);
                    Err(format!("Error status: {}", status))
                }
            }
            Err(e) => {
                println!("{:?}", e);
                Err(e.to_string())
            }
        };

        result.map_err(error)
    }

    ///
    /// Fields: vectors, namespace
    ///
    pub async fn upsert(&self) -> Result<PineconeResponse, PineconeApiError> {
        // vectors must not be empty
        if self.vectors().as_ref().map_or(true, |v| v.len() == 0) {
            return Err(PineconeApiError::UpsertError(
                "vectors cannot be empty".to_string(),
            ));
        }

        self.send(UPSERT, |error_message| {
                PineconeApiError::UpsertError(error_message)
            }).await
    }

    ///
    /// Fields: namespace, top_k, filter, include_values, include_metadata, vector, sparse_vector, id
    ///
    pub async fn query(&self) -> Result<PineconeResponse, PineconeApiError> {
        if let Some(value) = self.validate_query_request() {
            return value;
        }

        self.send(QUERY, |error_message| {
            PineconeApiError::QueryError(error_message)
        }).await
    }

    fn validate_query_request(&self) -> Option<Result<PineconeResponse, PineconeApiError>> {
        if self.id().as_ref().map_or(false, |id| id.len() > 512) {
            return Some(Err(PineconeApiError::QueryError(
                "id length must be 512 or less".to_string(),
            )));
        }

        if self.vector().is_none() {
            return Some(Err(PineconeApiError::QueryError(
                "vector cannot be empty".to_string(),
            )));
        }

        if self.top_k().is_none() {
            return Some(Err(PineconeApiError::QueryError(
                "top_k cannot be empty".to_string(),
            )));
        } else if self.top_k().as_ref().map_or(false, |k| k < &1) {
            return Some(Err(PineconeApiError::QueryError(
                "top_k must be at least 1".to_string(),
            )));
        }

        if let Some(sparse) = &self.sparse_vector() {
            let indices_len = sparse.indices().as_ref().map_or(0, |indices| indices.len());
            let values_len = sparse.values().len();

            if indices_len == 0 {
                return Some(Err(PineconeApiError::QueryError(
                    "indices cannot be empty when providing a sparse_vector".to_string(),
                )));
            } else if indices_len != values_len {
                return Some(Err(PineconeApiError::QueryError(
                    "indices and values must have the same length when providing a sparse_vector"
                        .to_string(),
                )));
            }
        }

        None
    }

    ///
    /// Fields: id, values, sparse_values, set_metadata, namespace
    ///
    pub async fn update(&self) -> Result<PineconeResponse, PineconeApiError> {
        if let Some(value) = self.validate_update_request() {
            return value;
        }

        self.send(UPDATE, |error_message| {
            PineconeApiError::UpdateError(error_message)
        })
            .await
    }

    fn validate_update_request(&self) -> Option<Result<PineconeResponse, PineconeApiError>> {
        let id_len = self.id().as_ref().map_or(0, |id| id.len());
        if id_len <= 1 || id_len > 512 {
            return Some(Err(PineconeApiError::UpdateError(
                "id is required and must have a length between 1 and 512".to_string(),
            )));
        }

        if self.sparse_values().is_none() && self.metadata().is_none() {
            return Some(Err(PineconeApiError::UpdateError(
                "You must provide something to update! Provide either a sparse_values or set_metadata field".to_string(),
            )));
        }

        if let Some(sparse) = &self.sparse_values() {
            let indices_len = sparse.indices().as_ref().map_or(0, |indices| indices.len());
            let values_len = sparse.values().len();

            if values_len == 0 || indices_len == 0 || (values_len != indices_len) {
                return Some(Err(PineconeApiError::UpdateError(
                    "sparse indices and values cannot be empty and must have the same length."
                        .to_string(),
                )));
            }
        }

        None
    }

    ///
    /// Fields: ids, namespace
    ///
    pub async fn fetch(&self) -> Result<PineconeResponse, PineconeApiError> {
        let url: String;
        if let Some(IdList::TextIds(ids)) = &self.ids() {
            let url_temp = ids
                .iter()
                .fold(BASE_URL.to_owned(), |url, id| format!("{}&ids={}", url, id));

            if let Some(namespace) = self.namespace() {
                url = format!("{}&namespace={}", url_temp, namespace);
            } else {
                url = url_temp;
            }
        } else {
            return Err(PineconeApiError::FetchError(
                "ids cannot be empty".to_string(),
            ));
        }

        let response = CLIENT
            .get(url)
            .send()
            .await
            .map_err(|e| PineconeApiError::FetchError(e.to_string()))?
            .json()
            .await
            .map_err(|e| PineconeApiError::FetchError(e.to_string()))?;

        Ok(response)
    }

    ///
    /// Fields: ids, delete_all, filter, namespace
    ///
    pub async fn delete(&self) -> Result<PineconeResponse, PineconeApiError> {
        if let Some(value) = self.validate_delete_request() {
            return value;
        }

        self.send(DELETE, |error_message| {
            PineconeApiError::DeleteError(error_message)
        })
            .await
    }

    fn validate_delete_request(&self) -> Option<Result<PineconeResponse, PineconeApiError>> {
        if self.ids().is_none() && self.delete_all().is_none() {
            return Some(Err(PineconeApiError::DeleteError(
                "You must provide either delete_all or ids to delete".to_string(),
            )));
        }

        match &self.ids() {
            Some(IdList::IntegerIds(_)) => {
                return Some(Err(PineconeApiError::DeleteError(
                    "ids must be Strings".to_string(),
                )));
            }
            Some(IdList::TextIds(val)) => {
                if val.len() == 0 {
                    return Some(Err(PineconeApiError::DeleteError(
                        "ids cannot be empty".to_string(),
                    )));
                }
            }
            None => {
                return None;
            }
        }
        None
    }
    // validation functions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::libs::{
        openai_api::OpenAIEmbeddingResponse, pinecone_data::Vector,
    };
    use serde_json::from_reader;
    use std::{fs::File, io::BufReader};
    use tokio::test;

    #[ignore]
    #[test]
    async fn test_upsert() {
        let embedding = read_openai_response_from_file("resources/embedding_example.json");

        let namespace = "test_namespace".to_string();
        let vectors = vec![Vector::builder()
            .id("dummy-id".to_string())
            .values(embedding)
            .build()];

        let response = PineconeRequest::builder()
            .vectors(vectors)
            .namespace(namespace)
            .build()
            .upsert()
            .await;

        let response = response.unwrap();
        println!("{:?}", response);
    }

    #[ignore]
    #[test]
    async fn test_query() {
        let embedding = read_openai_response_from_file("resources/embedding_example.json");

        let namespace = "test_namespace".to_string();

        let vectors = vec![Vector::builder()
            .id("dummy-id".to_string())
            .values(embedding)
            .build()];
    }

    #[ignore]
    #[test]
    async fn test_update() {
        todo!()
    }

    #[ignore]
    #[test]
    async fn test_fetch() {
        todo!()
    }

    #[ignore]
    #[test]
    async fn test_delete() {
        todo!()
    }

    fn read_openai_response_from_file(path: &str) -> Vec<f32> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let response: OpenAIEmbeddingResponse = from_reader(reader).unwrap();
        response.data().get(0).unwrap().embedding().to_owned()
    }
}

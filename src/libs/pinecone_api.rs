use lazy_static::lazy_static;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, sync::Arc};
use thiserror::Error;
use typed_builder::TypedBuilder;

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

fn headers(api_key: String) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Api-Key: {}", api_key).parse().unwrap(),
    );
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers
}

const BASE_URL: &str = "https://test-index-1a567db.svc.us-west4-gcp.pinecone.io/";
const UPSERT: &str = "vectors/upsert";
const QUERY: &str = "query";
const UPDATE: &str = "vectors/update";
const FETCH: &str = "vectors/fetch?ids=";
const DELETE: &str = "vectors/delete?ids=";

///
///
/// # Fields
///  
///
///
#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct PineconeRequest {
    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    vectors: Option<Vec<Vector>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vector>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "topK")]
    top_k: Option<u32>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "includeMetadata")]
    include_metadata: Option<bool>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "includeValues")]
    include_values: Option<bool>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "setMetadata")]
    set_metadata: Option<HashMap<String, String>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "sparseVector")]
    sparse_vector: Option<Vector>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "sparseValues")]
    sparse_values: Option<Vec<Vector>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    ids: Option<IdList>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<HashMap<String, String>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "deleteAll")]
    delete_all: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum IdList {
    IntegerIds(Vec<u32>),
    TextIds(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct Vector {
    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,

    values: Vec<f32>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    indices: Option<Vec<u32>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PineconeResponse {}

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

impl PineconeRequest {
    pub async fn upsert(&self) -> Result<PineconeResponse, PineconeApiError> {
        // vectors must not be empty
        if self.vectors.is_none() || self.vectors.as_ref().unwrap().len() == 0 {
            return Err(PineconeApiError::UpsertError(
                "vectors cannot be empty for upsert request.".to_string(),
            ));
        }

        let response: PineconeResponse = CLIENT
            .post(format!("{}{}", BASE_URL, UPSERT))
            .json(self)
            .send()
            .await
            .map_err(|e| PineconeApiError::UpsertError(e.to_string()))?
            .json()
            .await
            .map_err(|e| PineconeApiError::UpsertError(e.to_string()))?;

        Ok(response)
    }

    pub async fn query(&self) -> Result<PineconeResponse, PineconeApiError> {
        if self.vector.is_none() {
            return Err(PineconeApiError::QueryError(
                "vector cannot be empty".to_string(),
            ));
        }

        if self.top_k.is_none() {
            return Err(PineconeApiError::QueryError(
                "top_k cannot be empty".to_string(),
            ));
        }

        if self.sparse_vector.is_some() {
            let sparse = self.sparse_vector.as_ref().unwrap();
            if sparse.indices.is_none() {
                return Err(PineconeApiError::QueryError(
                    "indices cannot be empty when providing a sparse_vector".to_string(),
                ));
            } else if sparse.indices.as_ref().unwrap().len() != sparse.values.len() {
                return Err(PineconeApiError::QueryError(
                    "indices and values must have the same length when providing a sparse_vector"
                        .to_string(),
                ));
            }
        }

        let response: PineconeResponse = CLIENT
            .post(format!("{}{}", BASE_URL, QUERY))
            .json(self)
            .send()
            .await
            .map_err(|e| PineconeApiError::QueryError(e.to_string()))?
            .json()
            .await
            .map_err(|e| PineconeApiError::QueryError(e.to_string()))?;

        Ok(response)
    }

    pub async fn update(&self) {
        todo!()
    }

    pub async fn fetch(&self) {
        todo!()
    }

    pub async fn delete(&self) {
        todo!()
    }
}

// pub async fn semantic_search(
//     index_name: &str,
//     embedded_query: &str,
//     num_results: usize,
// ) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {

//     let results = CLIENT
//         .fetch(index_name, &query_embeddings, num_results)
//         .await?;

//     // Convert Pinecone results to a Vec<(String, f32)> format
//     let mut ranked_results: Vec<(String, f32)> = Vec::new();
//     for (id, score) in results {
//         ranked_results.push((id, score));
//     }

//     Ok(ranked_results)
// }

// fn extract_text_from_pdf(filename: &str) -> String {
//     let bytes = std::fs::read(filename).unwrap();
//     let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
//     out
// }

// pub fn process_files(
//     files: &[String],
//     max_tokens: u32,
// ) -> Result<Vec<Result<Vec<Vec<u32>>, String>>, Box<dyn std::error::Error>> {
//     let embeddings: Vec<Result<Vec<Vec<u32>>, String>> = files
//         // .par_iter()
//         .iter()
//         .map(|filename| {
//             let res = extract_text_from_pdf(filename);
//             match res {
//                 Ok(text) => match embed_text(&text, max_tokens) {
//                     Ok(embedding) => Ok(embedding),
//                     Err(e) => Err(format!("Error embedding text from {}: {}", filename, e)),
//                 },
//                 Err(e) => Err(format!("Error extracting text from {}: {}", filename, e)),
//             }
//         })
//         .collect();

//     Ok(embeddings)
// }

// fn process_texts(
//     text_array: &[String],
//     max_tokens: u32,
// ) -> Result<Vec<Result<Vec<Vec<f32>>, String>>, Box<dyn std::error::Error>> {
//     let embeddings: Vec<Result<Vec<Vec<f32>>, String>> = text_array
//         .iter()
//         .map(|text| match embed_text(text, max_tokens) {
//             Ok(embedding) => Ok(embedding),
//             Err(e) => Err(format!("Error embedding text: {}", e)),
//         })
//         .collect();

//     Ok(embeddings)
// }

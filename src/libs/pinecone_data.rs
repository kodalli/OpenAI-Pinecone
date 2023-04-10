use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typed_builder::TypedBuilder;

use super::database_api::Observer;

/// PineconeRequest represents a request to the Pinecone API.
///
/// # Fields
///  
/// * `vectors`: Optional list of vectors to store.
/// * `namespace`: Optional namespace for the request.
/// * `vector`: Optional single vector to store.
/// * `top_k`: Optional parameter for the number of nearest neighbors to return.
/// * `include_metadata`: Optional flag to include metadata in the response.
/// * `include_values`: Optional flag to include values in the response.
/// * `set_metadata`: Optional metadata to set for the specified vector.
/// * `sparse_vector`: Optional sparse vector representation.
/// * `sparse_values`: Optional sparse vector values.
/// * `ids`: Optional list of integer or text IDs for fetching vectors.
/// * `id`: Optional single ID for fetching a vector.
/// * `filter`: Optional filter for the request.
/// * `delete_all`: Optional flag to delete all data from the namespace.
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
    top_k: Option<i64>,

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
    sparse_values: Option<Vector>,

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

    // Used to update database with raw text data
    #[serde(skip)]
    #[builder(setter(strip_option), default)]
    observer: Option<Box<dyn Observer>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum IdList {
    IntegerIds(Vec<i64>),
    TextIds(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder, Clone)]
pub struct Vector {
    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,

    values: Vec<f32>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    indices: Option<Vec<i64>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PineconeResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    vectors: Option<Vec<AdditionalProp>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    matches: Option<Vec<Match>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    upserted_count: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AdditionalProp {
    id: String,
    values: Vec<f32>,
    metadata: HashMap<String, String>,

    #[serde(rename = "sparseValues")]
    sparse_values: Vector,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Match {
    id: String,
    score: f32,
    values: Vec<f32>,
    metadata: HashMap<String, String>,

    #[serde(rename = "sparseValues")]
    sparse_values: Vector,
}

impl PineconeRequest {
    pub fn vectors(&self) -> &Option<Vec<Vector>> {
        &self.vectors
    }

    pub fn namespace(&self) -> &Option<String> {
        &self.namespace
    }

    pub fn vector(&self) -> &Option<Vector> {
        &self.vector
    }

    pub fn top_k(&self) -> &Option<i64> {
        &self.top_k
    }

    pub fn include_metadata(&self) -> &Option<bool> {
        &self.include_metadata
    }

    pub fn include_values(&self) -> &Option<bool> {
        &self.include_values
    }

    pub fn set_metadata(&self) -> &Option<HashMap<String, String>> {
        &self.set_metadata
    }

    pub fn sparse_vector(&self) -> &Option<Vector> {
        &self.sparse_vector
    }

    pub fn sparse_values(&self) -> &Option<Vector> {
        &self.sparse_values
    }

    pub fn ids(&self) -> &Option<IdList> {
        &self.ids
    }

    pub fn id(&self) -> &Option<String> {
        &self.id
    }

    pub fn filter(&self) -> &Option<HashMap<String, String>> {
        &self.filter
    }

    pub fn delete_all(&self) -> &Option<bool> {
        &self.delete_all
    }

    pub fn observer(&self) -> &Option<Box<dyn Observer>> {
        &self.observer
    }
}

impl Vector {
    pub fn id(&self) -> &Option<String> {
        &self.id
    }

    pub fn values(&self) -> &Vec<f32> {
        &self.values
    }

    pub fn indices(&self) -> &Option<Vec<i64>> {
        &self.indices
    }

    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

impl AdditionalProp {
    pub fn id(&self) -> &String {
        &self.id
    }

    pub fn values(&self) -> &Vec<f32> {
        &self.values
    }

    pub fn sparse_values(&self) -> &Vector {
        &self.sparse_values
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

impl Match {
    pub fn id(&self) -> &String {
        &self.id
    }

    pub fn score(&self) -> f32 {
        self.score
    }

    pub fn values(&self) -> &Vec<f32> {
        &self.values
    }

    pub fn sparse_values(&self) -> &Vector {
        &self.sparse_values
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

impl PineconeResponse {
    pub fn vectors(&self) -> &Option<Vec<AdditionalProp>> {
        &self.vectors
    }

    pub fn namespace(&self) -> &Option<String> {
        &self.namespace
    }

    pub fn matches(&self) -> &Option<Vec<Match>> {
        &self.matches
    }

    pub fn upserted_count(&self) -> &Option<i64> {
        &self.upserted_count
    }
}

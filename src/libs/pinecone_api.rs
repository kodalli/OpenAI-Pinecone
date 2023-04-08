use lazy_static::lazy_static;
use reqwest::Client;
use serde_json::json;
use std::{env, sync::Arc};

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

pub async fn upsert_to_pinecone(
    vector_id: &str,
    vector_data: &[Vec<u32>],
) -> Result<(), Box<dyn std::error::Error>> {
    let vector_data_serialized = json!(vector_data);

    let payload = json!({
        "id": vector_id,
        "vector": vector_data_serialized,
    });

    let response = CLIENT
        .post(format!("{}{}", BASE_URL, UPSERT))
        .json(&payload)
        .send()
        .await?;

    if response.status().is_success() {
        println!("Upsert successful for vector ID: {}", vector_id);
    } else {
        println!("Upsert failed for vector ID: {}", vector_id);
    }

    Ok(())
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

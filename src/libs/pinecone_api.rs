use lazy_static::lazy_static;
use pdf_extract::extract_text;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use reqwest::Client;
use serde_json::json;
use std::{env, sync::Arc};
use tiktoken_rs::cl100k_base;

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

pub fn headers(api_key: String) -> reqwest::header::HeaderMap {
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

pub fn embed_text(
    text: &str,
    max_tokens: usize,
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let bpe = cl100k_base().unwrap();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut token_count = 0;

    for line in text.lines() {
        let line_tokens: Vec<&str> = line.split_whitespace().collect();
        let line_token_count = line_tokens.len();

        if token_count + line_token_count > max_tokens {
            // Embed the current chunk and clear it for the next chunk
            let chunk_embeddings = bpe.encode_with_special_tokens(&current_chunk);
            chunks.push(chunk_embeddings.into_iter().map(|x| x as u32).collect());
            current_chunk.clear();
            token_count = 0;
        }

        // Add the line to the current chunk
        current_chunk.push_str(line);
        current_chunk.push('\n');
        token_count += line_token_count;
    }

    // Embed the last chunk if it has any content
    if !current_chunk.is_empty() {
        let chunk_embeddings = bpe.encode_with_special_tokens(&current_chunk);
        chunks.push(chunk_embeddings.into_iter().map(|x| x as u32).collect());
    }

    Ok(chunks)
}

async fn upsert_to_pinecone(
    vector_id: &str,
    vector_data: &[Vec<usize>],
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
//     query: &str,
//     model: &Model,
//     num_results: usize,
// ) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
//     // Embed the query using the same model as the documents
//     let embedder = Embedder::new(model.clone());
//     let query_embeddings = embedder.embed_sentence(query)?;

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

fn process_files(
    files: &[String],
    max_tokens: usize,
) -> Result<Vec<Result<Vec<Vec<u32>>, String>>, Box<dyn std::error::Error>> {
    let embeddings: Vec<Result<Vec<Vec<u32>>, String>> = files
        .par_iter()
        .map(|filename| {
            let res = extract_text(filename);
            match res {
                Ok(text) => match embed_text(&text, max_tokens) {
                    Ok(embedding) => Ok(embedding),
                    Err(e) => Err(format!("Error embedding text from {}: {}", filename, e)),
                },
                Err(e) => Err(format!("Error extracting text from {}: {}", filename, e)),
            }
        })
        .collect();

    Ok(embeddings)
}

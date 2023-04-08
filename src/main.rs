mod libs;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read},
};

use libs::openai_api::{OpenAIEmbeddingRequest, OpenAIEmbeddingResponse};
use serde_json::{from_reader, to_writer};

use crate::libs::openai_api::{Message, OpenAIRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // test_openai_api().await?;
    // let prompt = "Example text for embedding.";
    // test_openai_embedding_api(prompt, "resources/embedding_example.json").await?;
    // openai_embedding_read("resources/embedding_example.json");
    let path = "resources/longer.txt";
    let output_file = "resources/longer_embedding.json";
    test_txt_file_to_embedding_api(path, output_file).await?;
    let embedding = openai_embedding_read(output_file);

    Ok(())
}

// From testing, all the embeddings regardless of size are 6kb
fn openai_embedding_read(path: &str) -> Vec<f32> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let response: OpenAIEmbeddingResponse = from_reader(reader).unwrap();
    let embedding = response.data().get(0).unwrap().embedding();
    // bytes
    let kb = calc_data_size(embedding);
    println!("{:?}", embedding);
    println!("takes: {} kb", kb);

    embedding.to_owned()
}

fn calc_data_size(embedding: &Vec<f32>) -> usize {
    let size = 4 * embedding.len() + 24;
    let kb = size / 1000;
    kb
}

async fn test_txt_file_to_embedding_api(
    path: &str,
    output_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    test_openai_embedding_api(&text, output_file).await?;
    Ok(())
}

async fn test_openai_embedding_api(
    prompt: &str,
    output_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = OpenAIEmbeddingRequest::builder()
        .model("text-embedding-ada-002".to_string())
        .input(prompt.to_string())
        .build()
        .send()
        .await?;

    let file = File::create(output_file).unwrap();
    let writer = BufWriter::new(file);
    to_writer(writer, &response).unwrap();
    println!("Generated embeddings: {:?}", response);

    Ok(())
}

async fn test_openai_api() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "How do you upload your mind to skynet?";

    let msg = Message::builder()
        .role("user".to_string())
        .content(prompt.to_string())
        .build();

    let tokens = msg.get_tokens()?;
    println!("Tokens: {}", tokens.len());

    let messages = vec![msg];

    let response = OpenAIRequest::builder()
        .model("gpt-3.5-turbo".to_string())
        .messages(messages)
        .build()
        .send()
        .await?;

    println!("Generated text: {:?}", response);
    Ok(())
}

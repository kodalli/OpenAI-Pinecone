use dotenv;
use reqwest::Client;
use std::env;
use tiktoken_rs::cl100k_base;

mod libs;
use crate::libs::openai_api::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let api_key = env::var("OPENAI_API_KEY")?;
    let prompt = "Censor the curse words in the following sentence and then display the uncensored text for reference: You're fucking crazy you goddamn lunatic!";
    let bpe = cl100k_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(prompt);
    println!("Tokens: {}", tokens.len());

    let client = Client::builder()
        .default_headers(headers(api_key))
        .build()?;

    let model = "gpt-3.5-turbo";
    let user = "user";

    let msg = Message::builder()
        .role(user.to_string())
        .content(prompt.to_string())
        .build();

    let messages = vec![msg];

    let request_data = OpenAIRequest::builder()
        .model(model.to_string())
        .messages(messages)
        .build();

    request_data.validate()?;

    // let request_data = OpenAIRequest {model, messages};

    let response: OpenAIResponse = client
        .post("https://api.openai.com/v1/chat/completions")
        .json(&request_data)
        .send()
        .await?
        .json()
        .await?;

    println!("Generated text: {:?}", response);

    Ok(())
}

fn headers(api_key: String) -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {}", api_key).parse().unwrap(),
    );
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers
}

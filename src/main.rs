mod libs;
use crate::libs::openai_api::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "Censor the curse words in the following sentence and then display the uncensored text for reference: You're fucking crazy you goddamn lunatic!";

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

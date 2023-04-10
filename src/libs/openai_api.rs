use lazy_static::lazy_static;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, error::Error, sync::Arc};
use tiktoken_rs::cl100k_base;
use typed_builder::TypedBuilder;

lazy_static! {
    static ref CLIENT: Arc<Client> = {
        dotenv::dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("Failed to locate api key.");

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
        format!("Bearer {}", api_key).parse().unwrap(),
    );
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers
}

/// Represents a request body for OpenAI's Embedding API.
///
/// # Fields
///
/// * `input`: Required. Input text to get embeddings for, encoded as a `String` or array of tokens.
/// * `model`: Required. ID of the model to use. Use the List models API to see available models or refer to the Model overview for descriptions.
/// * `user`: Optional. A unique identifier representing your end-user, which can help OpenAI monitor and detect abuse.
///
/// # Example
///
/// ```rust
/// let embedding_request = OpenAIEmbeddingRequest::builder()
///     .input("This is an example text.")
///     .model("text-embedding-ada-002")
///     .user(Some("unique_user_id"))
///     .build();
/// ```
#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct OpenAIEmbeddingRequest {
    input: String,
    model: String,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl OpenAIEmbeddingRequest {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        let tokens = get_tokens(&self.input)?;
        debug_assert!(tokens.len() <= 8191);
        Ok(())
    }

    pub async fn send(&self) -> Result<OpenAIEmbeddingResponse, Box<dyn Error>> {
        self.validate()?;

        let response: OpenAIEmbeddingResponse = CLIENT
            .post("https://api.openai.com/v1/embeddings")
            .json(self)
            .send()
            .await
            .map_err(|_| "Failed to send request.")?
            .json()
            .await
            .map_err(|_| "Failed to deserialize response.")?;

        Ok(response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIEmbeddingResponse {
    data: Vec<Embedding>,
    model: String,
    object: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Embedding {
    embedding: Vec<f32>,
    index: u32,
    object: String,
}

/// Represents a request body for OpenAI's Chat API.
///
/// # Fields
///
/// * `model`: Required. ID of the model to use (e.g., "gpt-3.5-turbo").
/// * `messages`: Required. An array of messages in the chat format.
/// * `temperature`: Optional. A number between 0 and 2 controlling output randomness. Higher values make output more random, lower values make it more focused.
/// * `top_p`: Optional. A number between 0 and 1 for nucleus sampling. Smaller values focus on top probable tokens.
/// * `n`: Optional. The number of chat completion choices to generate for each input message.
/// * `stream`: Optional. If set, partial message deltas will be sent.
/// * `stop`: Optional. Up to 4 sequences where the API will stop generating further tokens.
/// * `max_tokens`: Optional. The maximum number of tokens to generate in the chat completion.
/// * `presence_penalty`: Optional. A number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.
/// * `frequency_penalty`: Optional. A number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.
/// * `logit_bias`: Optional. A map to modify the likelihood of specified tokens appearing in the completion. Maps tokens to associated bias values from -100 to 100.
/// * `user`: Optional. A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.
///
/// # Example
///
/// ```rust
/// let chat_request = OpenAIRequest::builder()
///     .model("gpt-3.5-turbo")
///     .messages(vec![
///         "User: What is the capital of France?".to_string(),
///         "Assistant: The capital of France is Paris.".to_string(),
///     ])
///     .temperature(0.5)
///     .max_tokens(50)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<std::collections::HashMap<String, f64>>,

    #[builder(setter(strip_option), default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl OpenAIRequest {
    pub fn validate(&self) -> Result<(), OpenAIApiError> {
        match (
            self.temperature.unwrap_or(0.0),
            self.top_p.unwrap_or(0.0),
            self.presence_penalty.unwrap_or(0.0),
            self.frequency_penalty.unwrap_or(0.0),
        ) {
            (temp, _, _, _) if temp < 0.0 || temp > 2.0 => Err(OpenAIApiError::InvalidTemperature),
            (_, p, _, _) if p < 0.0 || p > 1.0 => Err(OpenAIApiError::InvalidTopP),
            (_, _, presence_penalty, _) if presence_penalty < -2.0 || presence_penalty > 2.0 => {
                Err(OpenAIApiError::InvalidPresencePenalty)
            }
            (_, _, _, frequency_penalty) if frequency_penalty < -2.0 || frequency_penalty > 2.0 => {
                Err(OpenAIApiError::InvalidFrequencyPenalty)
            }
            _ => Ok(()),
        }
    }

    pub async fn send(&self) -> Result<OpenAIResponse, Box<dyn Error>> {
        self.validate()?;

        let response: OpenAIResponse = CLIENT
            .post("https://api.openai.com/v1/chat/completions")
            .json(self)
            .send()
            .await
            .map_err(|_| "Failed to send request.")?
            .json()
            .await
            .map_err(|_| "Failed to deserialize response.")?;

        Ok(response)
    }
}

#[derive(Debug)]
pub enum OpenAIApiError {
    InvalidTemperature,
    InvalidTopP,
    InvalidPresencePenalty,
    InvalidFrequencyPenalty,
}

// Implement the std::error::Error trait for the ValidationError enum
impl std::error::Error for OpenAIApiError {}

// Implement the std::fmt::Display trait for the ValidationError enum
impl std::fmt::Display for OpenAIApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIApiError::InvalidTemperature => {
                write!(f, "temperature must be between 0 and 2.")
            }
            OpenAIApiError::InvalidTopP => write!(f, "Top_p must be between 0 and 1."),
            OpenAIApiError::InvalidPresencePenalty => {
                write!(f, "Presence_penalty must be between -2.0 and 2.0.")
            }
            OpenAIApiError::InvalidFrequencyPenalty => {
                write!(f, "Frequency_penalty must be between -2.0 and 2.0.")
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder, Clone)]
pub struct Message {
    role: String,
    content: String,
}

impl Message {
    pub fn to_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn get_tokens(&self) -> Result<Vec<usize>, serde_json::Error> {
        let msg = &self.to_string()?;
        let tokens = get_tokens(msg)?;
        Ok(tokens)
    }
}

pub fn get_tokens(msg: &str) -> Result<Vec<usize>, serde_json::Error> {
    let bpe = cl100k_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(msg);
    Ok(tokens)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    message: Message,
    finish_reason: String,
    index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    usage: Usage,
    choices: Vec<Choice>,
}

// getters
impl Message {
    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

impl Choice {
    pub fn message(&self) -> &Message {
        &self.message
    }

    pub fn finish_reason(&self) -> &str {
        &self.finish_reason
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

impl Usage {
    pub fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    pub fn completion_tokens(&self) -> Option<u32> {
        self.completion_tokens
    }

    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }
}

impl OpenAIResponse {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn object(&self) -> &str {
        &self.object
    }

    pub fn created(&self) -> u64 {
        self.created
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }
}

impl OpenAIEmbeddingResponse {
    pub fn data(&self) -> &Vec<Embedding> {
        &self.data
    }

    pub fn model(&self) -> &String {
        &self.model
    }

    pub fn object(&self) -> &String {
        &self.object
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }
}

impl Embedding {
    pub fn embedding(&self) -> &Vec<f32> {
        &self.embedding
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn object(&self) -> &String {
        &self.object
    }
}

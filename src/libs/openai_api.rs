use lazy_static::lazy_static;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, error::Error, sync::Arc};
use tiktoken_rs::{cl100k_base, CoreBPE};
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
    static ref BPE: CoreBPE = cl100k_base().unwrap();
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
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(temp) = self.temperature {
            if temp < 0.0 || temp > 2.0 {
                return Err(ValidationError::InvalidTemperature);
            }
        }

        if let Some(p) = self.top_p {
            if p < 0.0 || p > 1.0 {
                return Err(ValidationError::InvalidTopP);
            }
        }

        if let Some(penalty) = self.presence_penalty {
            if penalty < -2.0 || penalty > 2.0 {
                return Err(ValidationError::InvalidPresencePenalty);
            }
        }

        if let Some(penalty) = self.frequency_penalty {
            if penalty < -2.0 || penalty > 2.0 {
                return Err(ValidationError::InvalidFrequencyPenalty);
            }
        }

        Ok(())
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

#[derive(Debug)]
pub enum ValidationError {
    InvalidTemperature,
    InvalidTopP,
    InvalidPresencePenalty,
    InvalidFrequencyPenalty,
}

// Implement the std::error::Error trait for the ValidationError enum
impl std::error::Error for ValidationError {}

// Implement the std::fmt::Display trait for the ValidationError enum
impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::InvalidTemperature => {
                write!(f, "temperature must be between 0 and 2.")
            }
            ValidationError::InvalidTopP => write!(f, "Top_p must be between 0 and 1."),
            ValidationError::InvalidPresencePenalty => {
                write!(f, "Presence_penalty must be between -2.0 and 2.0.")
            }
            ValidationError::InvalidFrequencyPenalty => {
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
        let tokens = BPE.encode_with_special_tokens(msg);
        Ok(tokens)
    }
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
    completion_tokens: u32,
    total_tokens: u32,
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

    pub fn completion_tokens(&self) -> u32 {
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

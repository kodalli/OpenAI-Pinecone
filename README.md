## Plan

### ChatBot with Memory

### GPT-3.5

- Prefix prompt using personality and memories for chatbot agent
- Limit prompt to max gpt-3.5 tokens of 4096
- Keep track of entire conversation and append new message to prompt

### Pinecone

- Save vector buffers for context when chatting with different people
- Use semantic search to find relevant memories
- Create a context buffer that you can inject as part of the prompt.

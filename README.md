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

https://arxiv.org/abs/2304.03442v1
### Chatbot
- Memory stream
  - long-term memory list of experiences
  - retrieval model
  - relevance, recency, importance
- Reflection
  - synthesize memories -> higher level inferences over time
- Planning
  - high level action plan
  - act on conclusions
  - feed reflection and plans back into memory to influence future behavior
  
### Memory
- Paragraph seed memory, phrases semicolon delimited
- Can steer agent with inner voice
- environment status after interacting
- list of memory objects
  - description, creation timestamp, last accessed timestamp

#### Scoring Memory
- Recency
  - score memory objects higher if recently accessed
  - exponential decay function, decay factor 0.99
- Importance
  - high score for memories that are important
  - query model for integer scoring 
- Relevance
  - high score to memory that relevant
  - use embedding vector and cosine similarity
- min-max scale [0, 1]
- score = b0\*recency + b1\*importance + b2\*relevance
- include top ranked memories that fit in the context window to include in the prompt
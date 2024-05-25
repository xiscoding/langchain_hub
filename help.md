# LangChain Model Set UP
## OpenAI API models (gpt-x)
See langchain_manplus.py for example
1. login -> api tokens -> get/create new token
2. Set token in .env file
    1. OPENAI_API_KEY=
NOTE: GPT4 costs ~3cents per 1000 tokens (~1000 words) this includes all text, prompts, and completion responses.
## HUGGING FACE MODELS
blog: https://huggingface.co/blog/Andyrasika/agent-helper-langchain-hf
EXAMPLE FILE: langchain_huggingface_agent.py
1. hugging face read (read-only access) token
    1. settings -> access tokens
    2. New token -> type read
1. hugging face write (create or update content) token
- same as above, but with write selected
2. Set token in .env file
3. set model_id to desired model (check Use in Transformers link on hugging face for model="{model_id path}")
    1. model_id = "meta-llama/Meta-Llama-3-8B" (HUGGINGFACE llama 3 DOES NOT WORK WITH LANGCHAIN)
    2. other llama 3 models: [llama3_hub](https://github.com/xiscoding/llama3_hub) repo
    2. Models from hugging face are still subject to V/RAM requirements
### HugginFaceEndpoint
Docs: https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_endpoint.HuggingFaceEndpoint.html
Create a new model by parsing and validating input data from keyword arguments. <br>
To use this class, you should have installed the huggingface_hub package, and the environment variable HUGGINGFACEHUB_API_TOKEN set with your API token, or given as a named parameter to the constructor.
### ChatHuggingFace
Docs: https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.huggingface.ChatHuggingFace.html
Wrapper for using Hugging Face LLM’s as ChatModels.<br>
Works with HuggingFaceTextGenInference, **HuggingFaceEndpoint**, and HuggingFaceHub LLMs.<br>
Upon instantiating this class, the **model_id is resolved from the url provided to the LLM**, and the **appropriate tokenizer is loaded from the HuggingFace Hub**.

### LLAMA 3 inference
**Llama 3 is too big for HF inference**
#### option 1: OLLAMA
Docs: https://github.com/ollama/ollama
1. Install: curl -fsSL https://ollama.com/install.sh | sh
2. run and chat: 
    1. ollama run llama3
    2. USE ChatOllama with langchain
        - docs: https://python.langchain.com/docs/integrations/chat/ollama/
3. `ollama pull llama3`
    - This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model
#### option 2: local server
- Use webgui to get server then use that server to make requests 
    - server will be in format http://127.0.0.1:7860/
##### LM Studio Local Inference Server
* easiest way to get local server to run 
* URL: http://localhost:1234/v1/chat/completions

# LangChain Agent/LLM set up
## LLM vs Agent
* **Agent** is responsible for generating prompts, deciding when to generate a new one and passing it to the LLM.
* **LLM** generates responses based off of these prompts.<br>
See llama3_agent.py for example

Retrieval
LangGraph
RAG

# Tools
## SerpApi
Scrape Google and other search engines from our fast, easy, and complete API.<br> 
[SerpApi Documentation](https://serpapi.com/)<br>
### Google Search Engine Results API
Docs (No installation info): https://serpapi.com/search-api
`pip install google-search-results`<br>
required for `from langchain_community.utilities import SerpAPIWrapper`
## LLMMathChain
Docs: https://api.python.langchain.com/en/latest/chains/langchain.chains.llm_math.base.LLMMathChain.html <br>
Requires: `pip install numexpr`
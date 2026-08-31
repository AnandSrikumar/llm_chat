# LLM Chat

FastAPI chat application using Ollama for generation and vision, PostgreSQL with pgvector for storage, and Sentence Transformers for embeddings.

## Models

| Purpose | Model | Approx. download size |
| --- | --- | ---: |
| Chat | `ministral-3:3b` | 3.0 GB |
| Chat alternative | `qwen3:4b` | 3.2 GB |
| Vision | `qwen2.5vl:3b` | 3.2 GB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB |

`ministral-3:3b` is the chat model currently selected in `.env`. `qwen3:4b` is the Qwen 3 model supported by the application; `qwen3:3b` is not an Ollama tag used by this project. The embedding model downloads automatically from Hugging Face on the first application start.

## Download Ollama models

Install and start [Ollama](https://ollama.com/download), then run:

```bash
ollama pull ministral-3:3b
ollama pull qwen3:4b
ollama pull qwen2.5vl:3b
```

Set `OLLAMA_CHAT_MODEL` and `OLLAMA_VISION_MODEL` in `.env` to the models you want to use.

## Run in development

Start PostgreSQL:

```bash
docker compose up -d
```

Install Python packages and run FastAPI with reload enabled:

```bash
python -m pip install -r requirements.txt
fastapi dev main.py
```

The API is available at `http://localhost:8000`; interactive docs are at `http://localhost:8000/docs`.

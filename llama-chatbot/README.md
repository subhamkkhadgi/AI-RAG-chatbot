# Llama Chatbot

A production-ready **Streamlit** chatbot powered by **Llama 3.1** through **Ollama** for local inference, with a clean abstraction layer for adding cloud providers like Groq.

---

## Overview

This project provides a modular, testable chatbot application with a clear separation of concerns:

- **Streamlit** renders the UI (chat interface + sidebar controls).
- **ChatService** orchestrates the conversation flow (input validation, message history, streaming).
- **BaseLLMProvider** defines an abstract contract; concrete implementations (e.g., `OllamaProvider`)
  handle the actual inference call.
- **Pydantic models** represent messages, conversations, and provider requests.
- **Pydantic Settings** validates environment configuration at startup.

The architecture makes it straightforward to add new LLM providers (Groq, OpenAI, Anthropic, …)
without touching the UI or the core chat service.

---

## Features

- **Streamlit chat UI** — `st.chat_message`, `st.chat_input`, streaming output.
- **Local LLM inference** — runs Llama 3.1 via Ollama on your machine.
- **Provider abstraction** — `BaseLLMProvider` ABC with a registry/factory pattern.
- **Streaming responses** — tokens appear in real time as the model generates them.
- **Configurable models** — change the model name, temperature, and max tokens at runtime.
- **Conversation memory** — full chat history preserved in session state.
- **Runtime settings** — sidebar controls for provider, model, temperature, max tokens,
  system prompt, and a clear-chat button.
- **Error handling** — user-friendly safe messages; detailed errors in logs.
- **Testing** — 50+ unit tests with zero external dependencies (mocked providers, mocked Streamlit).

---

## Architecture

### Layer Diagram

```
┌──────────────────────────────────────────┐
│               User (Browser)              │
└────────────────┬─────────────────────────┘
                 │  HTTP (Streamlit)
┌────────────────▼─────────────────────────┐
│           app.py  (Entry Point)           │
│  st.set_page_config, init_session_state   │
└──────┬─────────────────────────┬──────────┘
       │                         │
┌──────▼──────────┐    ┌────────▼──────────┐
│   ui/sidebar    │    │ ui/chat_interface │
│  Controls       │    │  Messages + Input │
│  + Status       │    │  + Error display  │
└──────┬──────────┘    └────────▲──────────┘
       │                        │
       │            ┌───────────┴──────────┐
       │            │ services/chat_service │
       │            │  validates input      │
       └───────────►│  owns Conversation    │
                    │  creates ChatRequest  │
                    │  calls provider.chat  │
                    │  streams + stores     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  providers/factory    │
                    │  register_provider    │
                    │  create_provider      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   BaseLLMProvider     │
                    │   (Abstract Base)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   OllamaProvider      │
                    │   (concrete impl.)    │
                    └───────────┬───────────┘
                                │
                        ┌───────▼───────┐
                        │  Ollama Server │
                        │  (localhost)   │
                        └───────┬───────┘
                                │
                        ┌───────▼───────┐
                        │  Llama 3.1    │
                        │  (model)      │
                        └───────────────┘
```

### Data Flow

1. **User types a message** → captured by `ui/chat_interface.py` via `st.chat_input`.
2. UI calls `ChatService.stream_message(conversation, content)`.
3. **ChatService** validates input, adds a `ChatMessage(role=user)` to `Conversation`,
   builds a `ChatRequest`, and calls `provider.chat(request)`.
4. **OllamaProvider** (or any registered provider) sends the request to Ollama and
   yields streaming text chunks.
5. **ChatService** yields each chunk back to the UI, which renders it via `st.empty()`.
6. After streaming completes, **ChatService** adds the assistant message to `Conversation`.
7. **Sidebar** controls (provider, model, temperature) are stored in `st.session_state`
   and read by `ChatService` on each new message.

### Key Design Decisions

- **Only ChatService modifies `Conversation`.** The UI never calls `add_user_message`
  or `add_assistant_message` directly.
- **Providers are interchangeable.** Adding a new provider requires only a new class
  implementing `BaseLLMProvider` and one `register_provider()` call.
- **Streamlit is isolated to the UI layer.** Services, models, providers, and config
  never import Streamlit.
- **Provider SDKs (ollama) are isolated to their provider class.** The UI, services,
  and other providers never import them.

---

## Folder Structure

```
llama-chatbot/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Documented environment template
├── .gitignore                      # Git exclusion rules
├── README.md                       # This file
├── pyproject.toml                  # Ruff + pytest configuration
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic BaseSettings
│   ├── exceptions.py               # Exception hierarchy
│   ├── logging_config.py           # Application logger setup
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat.py                 # ChatMessage, Conversation, ChatRequest
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseLLMProvider (ABC)
│   │   ├── ollama_provider.py      # Ollama implementation
│   │   └── factory.py              # Provider registry + factory
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── system_prompts.py       # Default system prompt
│   ├── services/
│   │   ├── __init__.py
│   │   └── chat_service.py         # Chat orchestration
│   └── ui/
│       ├── __init__.py
│       ├── sidebar.py              # Sidebar controls
│       └── chat_interface.py       # Chat display + input
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures
    ├── prompts/
    │   ├── __init__.py
    │   └── test_system_prompts.py
    ├── providers/
    │   └── test_ollama_provider.py
    ├── services/
    │   ├── __init__.py
    │   └── test_chat_service.py
    └── ui/
        ├── __init__.py
        ├── test_sidebar.py
        └── test_chat_interface.py
```

---

## Requirements

- **Python 3.11+**
- **Ollama** (for local inference) — download from [ollama.com](https://ollama.com)
- **Operating system** — Windows 10/11, macOS, or Linux

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd llama-chatbot
```

### 2. Create a virtual environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com).

### 2. Pull the Llama 3.1 model

```bash
ollama pull llama3.1:8b
```

Other compatible models (no code changes required):

| Model | Command | Notes |
|-------|---------|-------|
| Llama 3.1 8B | `ollama pull llama3.1:8b` | Recommended — good balance of speed and quality |
| Llama 3.2 3B | `ollama pull llama3.2:3b` | Faster, less capable |
| Qwen 2.5 7B | `ollama pull qwen2.5:7b` | Strong alternative |

### 3. Verify Ollama is running

```bash
ollama list
```

If the model appears in the list, Ollama is ready.

---



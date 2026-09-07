# Renovation Calculator AI Assistant

## Table of Contents

- [Русский](#русский)
  - [О проекте](#о-проекте)
  - [Возможности](#возможности)
  - [Архитектура](#архитектура)
  - [Пайплайн данных](#пайплайн-данных)
  - [Структура проекта](#структура-проекта)
  - [Установка и запуск](#установка-и-запуск)
  - [Переменные окружения](#переменные-окружения)
  - [Основные команды](#основные-команды)
  - [Продакшен](#продакшен)
- [English](#english)
  - [About](#about)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Data Pipeline](#data-pipeline)
  - [Project Structure](#project-structure)
  - [Setup and Run](#setup-and-run)
  - [Environment Variables](#environment-variables)
  - [Main Commands](#main-commands)
  - [Production](#production)

---

## Русский

### О проекте

`Renovation Calculator AI Assistant` - это backend и data pipeline для AI-помощника в мобильном приложении по расчету ремонта квартир.

Ассистент отвечает на вопросы пользователей о ремонте, примерной стоимости работ, услугах компании, контактах и условиях. Для ответов используется RAG-подход: система ищет релевантные фрагменты в базе знаний и передает их языковой модели как контекст.

Мобильное приложение доступно в App Store: [Калькулятор ремонта](https://apps.apple.com/us/app/%D0%BA%D0%B0%D0%BB%D1%8C%D0%BA%D1%83%D0%BB%D1%8F%D1%82%D0%BE%D1%80-%D1%80%D0%B5%D0%BC%D0%BE%D0%BD%D1%82%D0%B0/id6761184107).

- данные берутся из реального сайта и JSON-прайса;
- backend отдает простой `/chat` API для мобильных клиентов;
- история переписки не хранится на стороне backend;
- входной пользовательский запрос ограничен по длине;
- векторное хранилище вынесено в Qdrant.

### Возможности

- Парсинг страниц сайта из sitemap.
- Очистка HTML от меню, popup, форм, footer/header и технического мусора.
- Загрузка структурированного прайса из `db.json`.
- Преобразование данных в `llama_index.core.Document`.
- Chunking документов через `SentenceSplitter`.
- Генерация embeddings через OpenAI-compatible API.
- Поддержка OpenAI и Yandex AI Studio compatible API.
- Загрузка векторов в Qdrant.
- Поиск релевантных чанков по пользовательскому вопросу.
- RAG-ответы через FastAPI endpoint `/chat`.
- Быстрый режим ответа `answer_fast()` для снижения стоимости запросов.
- CLI-скрипты для сборки документов, чанков, embeddings, Qdrant upload и тестов.

### Архитектура

```text
Website sitemap + db.json
        |
        v
Crawler / Pricing Loader
        |
        v
HTML Parser + Structured Documents
        |
        v
LlamaIndex Documents
        |
        v
Chunks
        |
        v
Embeddings
        |
        v
Qdrant Vector Store
        |
        v
FastAPI /chat endpoint
        |
        v
iOS / Android application
```

Ключевые компоненты:

- `app/crawler.py` - discovery URL из sitemap.
- `app/parser.py` - парсинг и очистка HTML-страниц.
- `app/pricing_loader.py` - загрузка и преобразование JSON-прайса в документы.
- `app/chunker.py` - разбиение документов на чанки.
- `app/embeddings.py` - клиент для embeddings.
- `app/vector_store.py` - работа с Qdrant.
- `app/rag.py` - RAG-логика, классификация, retrieval, генерация ответа.
- `app/api.py` - FastAPI backend.

### Пайплайн данных

1. `SitemapCrawler` получает sitemap index и выбирает нужные sitemap-файлы.
2. `SiteParser` очищает HTML, извлекает `title`, `h1`, headings, основной текст и metadata.
3. `PricingLoader` загружает `db.json` и превращает каждую услугу из прайса в отдельный документ.
4. `build_llama_docs.py` объединяет документы сайта и прайса в LlamaIndex documents.
5. `build_chunks.py` режет документы на чанки.
6. `build_embeddings.py` создает embeddings для чанков.
7. `upload_to_qdrant.py` пересоздает коллекцию Qdrant и загружает embedded chunks.
8. `RenovationRAG` принимает вопрос, получает embedding запроса, ищет релевантные chunks в Qdrant и генерирует ответ.
9. `FastAPI` endpoint `/chat` возвращает ответ мобильному приложению.

### Структура проекта

```text
.
├── app/
│   ├── api.py              # FastAPI application
│   ├── crawler.py          # Sitemap discovery
│   ├── parser.py           # HTML parsing and cleaning
│   ├── pricing_loader.py   # JSON pricing loader
│   ├── estimate_loader.py  # Excel estimate loader
│   ├── estimate_calculator.py # Estimate-based calculations
│   ├── chunker.py          # LlamaIndex document chunking
│   ├── embeddings.py       # Embedding client
│   ├── vector_store.py     # Qdrant wrapper
│   └── rag.py              # RAG orchestration
├── scripts/
│   ├── build_llama_docs.py
│   ├── build_chunks.py
│   ├── build_embeddings.py
│   ├── build_estimate_benchmarks.py
│   ├── upload_to_qdrant.py
│   ├── ask_rag.py
│   ├── evaluate_rag.py
│   └── tokens_counter.py
├── notebooks/
│   ├── parser_experiments.ipynb
│   ├── price_loader_experiments.ipynb
│   ├── estimate_loader_experiment.ipynb
│   ├── chunks_experiments.ipynb
│   ├── retrieval_experiments.ipynb
│   └── rag_experiments.ipynb
├── data/
│   └── cleaned/            # Generated local artifacts, not for public repo
└── requirements.txt
```

### Установка и запуск

Создание окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Локальный запуск API:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Проверка health endpoint:

```bash
curl http://localhost:8000/health
```

Пример запроса к чату:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Сколько примерно стоит ремонт квартиры 40 квадратов?"}'
```

### Переменные окружения

Пример `.env`:

```env
QDRANT_COLLECTION=renovation_docs
QDRANT_URL=https://your-qdrant-host:6333
QDRANT_API_KEY=your_qdrant_api_key

YANDEX_API_KEY=your_yandex_api_key
YANDEX_FOLDER_ID=your_folder_id
YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
YANDEX_CHAT_MODEL=gpt://your_folder_id/aliceai-llm-flash
YANDEX_DOC_EMBEDDING_MODEL=emb://your_folder_id/text-search-doc/latest
YANDEX_QUERY_EMBEDDING_MODEL=emb://your_folder_id/text-search-query/latest

OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

В проекте используются OpenAI-compatible клиенты, поэтому backend может работать как с OpenAI, так и с совместимыми API-провайдерами.

### Основные команды

Сборка документов:

```bash
python3 -m scripts.build_llama_docs
```

Создание чанков:

```bash
python3 -m scripts.build_chunks
```

Подсчет токенов:

```bash
python3 -m scripts.tokens_counter
```

Создание embeddings:

```bash
python3 -m scripts.build_embeddings
```

Загрузка в Qdrant:

```bash
python3 -m scripts.upload_to_qdrant
```

CLI-запрос к RAG:

```bash
python3 -m scripts.ask_rag "Сколько примерно стоит ремонт квартиры 40 квадратов?" --debug
```

Оценка на списке вопросов:

```bash
python3 -m scripts.evaluate_rag data/eval/questions.txt
```

### Продакшен

В production setup backend запускается как FastAPI-сервис за reverse proxy.

Рекомендуемая схема:

- FastAPI + Uvicorn на VPS.
- Nginx как reverse proxy.
- HTTPS через Let's Encrypt.
- Qdrant как managed/self-hosted vector database.
- API доступен мобильным приложениям через endpoint `/chat`.

Пример production endpoint:

```text
POST https://your-domain.com/chat
```

Ответ:

```json
{
  "answer": "Примерная стоимость ремонта квартиры площадью 40 м² — от 600 000 ₽...",
  "intent": "general_rag",
  "sources_count": 3
}
```

---

## English

### About

`Renovation Calculator AI Assistant` is a backend and data pipeline for an AI assistant integrated into a renovation cost calculator mobile application.

The assistant answers user questions about apartment renovation, approximate pricing, company services, contacts, payment terms, and related topics. It uses a RAG architecture: relevant knowledge base chunks are retrieved from a vector database and passed to a language model as context.

The mobile application is available on the App Store: [Renovation Calculator](https://apps.apple.com/us/app/%D0%BA%D0%B0%D0%BB%D1%8C%D0%BA%D1%83%D0%BB%D1%8F%D1%82%D0%BE%D1%80-%D1%80%D0%B5%D0%BC%D0%BE%D0%BD%D1%82%D0%B0/id6761184107).

The project is designed for a real commercial product:

- source data comes from a real website and a structured JSON price list;
- the backend exposes a simple `/chat` API for mobile clients;
- chat history is not stored by the backend;
- user input length is limited;
- vector search is handled by Qdrant.

### Features

- Website parsing through sitemap discovery.
- HTML cleaning from menus, popups, forms, headers, footers, and technical noise.
- Structured pricing ingestion from `db.json`.
- Conversion to `llama_index.core.Document`.
- Document chunking with `SentenceSplitter`.
- Embedding generation through an OpenAI-compatible API.
- Support for OpenAI and Yandex AI Studio compatible APIs.
- Vector storage in Qdrant.
- Relevant chunk retrieval by user query.
- RAG answer generation through FastAPI.
- Cost-optimized `answer_fast()` mode.
- CLI scripts for document generation, chunking, embeddings, Qdrant upload, and evaluation.

### Architecture

```text
Website sitemap + db.json
        |
        v
Crawler / Pricing Loader
        |
        v
HTML Parser + Structured Documents
        |
        v
LlamaIndex Documents
        |
        v
Chunks
        |
        v
Embeddings
        |
        v
Qdrant Vector Store
        |
        v
FastAPI /chat endpoint
        |
        v
iOS / Android application
```

Core modules:

- `app/crawler.py` - sitemap URL discovery.
- `app/parser.py` - HTML parsing and cleanup.
- `app/pricing_loader.py` - pricing JSON ingestion.
- `app/chunker.py` - document chunking.
- `app/embeddings.py` - embedding API client.
- `app/vector_store.py` - Qdrant wrapper.
- `app/rag.py` - RAG routing, retrieval, and answer generation.
- `app/api.py` - FastAPI backend.

### Data Pipeline

1. `SitemapCrawler` reads sitemap index and selects allowed sitemap files.
2. `SiteParser` cleans HTML and extracts `title`, `h1`, headings, main text, and metadata.
3. `PricingLoader` reads `db.json` and converts each pricing item into a document.
4. `build_llama_docs.py` combines website and pricing documents into LlamaIndex documents.
5. `build_chunks.py` splits documents into chunks.
6. `build_embeddings.py` creates embeddings for every chunk.
7. `upload_to_qdrant.py` recreates the Qdrant collection and uploads embedded chunks.
8. `RenovationRAG` receives a question, embeds it, retrieves relevant Qdrant chunks, and generates an answer.
9. FastAPI `/chat` returns the answer to mobile clients.

### Project Structure

```text
.
├── app/
│   ├── api.py              # FastAPI application
│   ├── crawler.py          # Sitemap discovery
│   ├── parser.py           # HTML parsing and cleaning
│   ├── pricing_loader.py   # JSON pricing loader
│   ├── estimate_loader.py  # Excel estimate loader
│   ├── estimate_calculator.py # Estimate-based calculations
│   ├── chunker.py          # LlamaIndex document chunking
│   ├── embeddings.py       # Embedding client
│   ├── vector_store.py     # Qdrant wrapper
│   └── rag.py              # RAG orchestration
├── scripts/
│   ├── build_llama_docs.py
│   ├── build_chunks.py
│   ├── build_embeddings.py
│   ├── build_estimate_benchmarks.py
│   ├── upload_to_qdrant.py
│   ├── ask_rag.py
│   ├── evaluate_rag.py
│   └── tokens_counter.py
├── notebooks/
│   ├── parser_experiments.ipynb
│   ├── price_loader_experiments.ipynb
│   ├── estimate_loader_experiment.ipynb
│   ├── chunks_experiments.ipynb
│   ├── retrieval_experiments.ipynb
│   └── rag_experiments.ipynb
├── data/
│   └── cleaned/            # Generated local artifacts, not for public repo
└── requirements.txt
```

### Setup and Run

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the local API:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Chat request example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How much does a 40 square meter apartment renovation cost?"}'
```

### Environment Variables

Example `.env`:

```env
QDRANT_COLLECTION=renovation_docs
QDRANT_URL=https://your-qdrant-host:6333
QDRANT_API_KEY=your_qdrant_api_key

YANDEX_API_KEY=your_yandex_api_key
YANDEX_FOLDER_ID=your_folder_id
YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
YANDEX_CHAT_MODEL=gpt://your_folder_id/aliceai-llm-flash
YANDEX_DOC_EMBEDDING_MODEL=emb://your_folder_id/text-search-doc/latest
YANDEX_QUERY_EMBEDDING_MODEL=emb://your_folder_id/text-search-query/latest

OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

The project uses OpenAI-compatible clients, so the backend can work with OpenAI or compatible providers.

### Main Commands

Build documents:

```bash
python3 -m scripts.build_llama_docs
```

Create chunks:

```bash
python3 -m scripts.build_chunks
```

Count tokens:

```bash
python3 -m scripts.tokens_counter
```

Build embeddings:

```bash
python3 -m scripts.build_embeddings
```

Upload to Qdrant:

```bash
python3 -m scripts.upload_to_qdrant
```

Ask through CLI:

```bash
python3 -m scripts.ask_rag "How much does a 40 square meter apartment renovation cost?" --debug
```

Evaluate on a question list:

```bash
python3 -m scripts.evaluate_rag data/eval/questions.txt
```

### Production

In production, the backend runs as a FastAPI service behind a reverse proxy.

Recommended setup:

- FastAPI + Uvicorn on a VPS.
- Nginx as a reverse proxy.
- HTTPS via Let's Encrypt.
- Qdrant as a managed or self-hosted vector database.
- Mobile clients communicate with the `/chat` endpoint.

Example production endpoint:

```text
POST https://your-domain.com/chat
```

Response:

```json
{
  "answer": "The approximate cost of renovating a 40 m² apartment starts from 600,000 RUB...",
  "intent": "general_rag",
  "sources_count": 3
}
```

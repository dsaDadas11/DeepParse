# DeepParse

DeepParse is a full-stack document parsing and analysis application. It features a React-based frontend and a Python backend utilizing FastAPI, Elasticsearch, PostgreSQL, and Redis.

## Architecture

* **Frontend:** React, Vite, TypeScript, TailwindCSS
* **Backend:** FastAPI (Python)
* **Database:** PostgreSQL
* **Search/Storage:** Elasticsearch
* **Cache:** Redis

## Setup and Installation

### Prerequisites

* Docker and Docker Compose
* Node.js (for local frontend development)

### Running the application via Docker

The entire stack can be launched using Docker Compose.

```bash
cd backend
docker-compose up -d
```

This will start the following services:
* FastAPI backend (Port 8000)
* Elasticsearch (es01)
* PostgreSQL (gsk_pg)
* Redis

### Local Frontend Development

```bash
cd frontend
npm install
npm run dev
```

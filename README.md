# DeepParse

DeepParse is a full-stack application designed for document parsing and intelligent analysis. It integrates a responsive frontend with a robust Python backend to process, search, and manage document knowledge efficiently.

## 🌟 Architecture & Tech Stack

*   **Frontend:** React 18, Vite, TypeScript, TailwindCSS (assumed), Ant Design (UI Library)
*   **Backend:** FastAPI (Python 3.x), Uvicorn
*   **Database:** PostgreSQL 15
*   **Search Engine:** Elasticsearch 8.11
*   **Caching & Queue:** Redis 7
*   **AI/ML Integration:** Sentence-Transformers, LlamaIndex, OpenAI API, XGBoost, and more.

## 🚀 Features (Overview)

*   **Document Parsing:** Extracts text and content from various formats (PDF, DOCX, TXT, MD, HTML, Excel, PPTX).
*   **Knowledge Base Management:** Stores and organizes parsed documents.
*   **Intelligent Search:** Utilizes Elasticsearch and embedding models for semantic and keyword search.
*   **Chat Interface:** Interactive chat system with access to document history and context.

## 🛠️ Setup and Installation

### Prerequisites

*   **Docker and Docker Compose** (Recommended for full stack deployment)
*   **Node.js 18+** (For local frontend development)
*   **Python 3.9+** (For local backend development)

### Running with Docker (Recommended)

The easiest way to get the entire application running is by using Docker Compose. This will spin up the backend API, PostgreSQL, Elasticsearch, and Redis.

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Ensure you have a `.env` and `app/key.txt` file configured correctly based on your environment needs.
3.  Start the services:
    ```bash
    docker-compose up -d
    ```

**Services Started:**
*   FastAPI Backend: `http://localhost:8000`
*   Elasticsearch: `es01` (Internal network)
*   PostgreSQL: `gsk_pg` (Internal network)
*   Redis: `gsk_redis` (Internal network)

### Local Development

#### Frontend

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be accessible at `http://localhost:5181` (default Vite port as configured).

#### Backend (Local Python Environment)

*It's highly recommended to use a virtual environment.*

1.  Navigate to the `backend/app` directory:
    ```bash
    cd backend/app
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    uvicorn app_main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *Note: You must have local instances or accessible URIs for PostgreSQL, Elasticsearch, and Redis configured in your environment variables for the backend to start successfully.*

## 📁 Project Structure

*   `/backend` - Contains the FastAPI application, database models, Docker configuration, and AI logic.
    *   `/app/router` - API route definitions (chat, history, etc.).
    *   `/app/service` - Business logic and document operations.
    *   `/app/database` - Database connections and operations.
*   `/frontend` - Contains the React web application.
    * `/src` - Source code, React components, state management (Valtio), 和 routing.

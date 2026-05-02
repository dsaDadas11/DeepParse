from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import chat_rt, history_rt
from runtime_config import APP_ROOT_PATH, CORS_ORIGINS, validate_runtime_config

validate_runtime_config()

app = FastAPI(root_path=APP_ROOT_PATH)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(chat_rt.router)
app.include_router(history_rt.router)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)

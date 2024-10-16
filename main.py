import os
import logging
import uvicorn

from fastapi import FastAPI
from app.routers.chat import chat_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import llama_index.core

load_dotenv()

app = FastAPI()


os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=8db46cf062333a78acb:6ac483c"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=8db46cf062333a78acb:6ac483c"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)

environment = os.getenv("ENVIRONMENT", "dev")
logger = logging.getLogger("uvicorn")

if environment == "dev":
    logger.warning(
        "Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Redirect to documentation page when accessing base URL
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

app.include_router(chat_router, prefix='/api/chat')

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    reload = True if environment == "dev" else False

    uvicorn.run(app="main:app", host=app_host, port=app_port, reload=reload)

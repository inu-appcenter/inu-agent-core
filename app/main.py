from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1.router import api_v1_router
from app.tools.registry import tool_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
    logger.info(f"Target LLM: {settings.LLM_MODEL_NAME} at {settings.LLM_BASE_URL}")

    # Synchronize OpenAPI tools from inu-portal-server
    try:
        synced_count = await tool_registry.sync_inu_portal_tools()
        logger.info(f"Tool registry initialized with {synced_count} active tools.")
    except Exception as e:
        logger.error(f"Error during tool registry synchronization: {e}", exc_info=True)

    yield

    # Shutdown actions
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title="INU Agent Core",
    description="Central AI Agent Orchestration Hub for Incheon National University AppCenter",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG or settings.APP_ENV != "production" else None,
    redoc_url="/redoc" if settings.DEBUG or settings.APP_ENV != "production" else None,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list if settings.cors_origins_list else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(api_v1_router, prefix="/api")


@app.get("/", summary="Root index")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": "0.1.0",
        "docs": "/docs" if settings.DEBUG or settings.APP_ENV != "production" else "disabled",
        "health": "/api/v1/health",
        "tools_count": len(tool_registry.list_tools()),
    }

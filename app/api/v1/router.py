from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.chat import router as chat_router

api_v1_router = APIRouter(prefix="/v1")

api_v1_router.include_router(health_router, tags=["Health"])
api_v1_router.include_router(chat_router, tags=["Chat"])

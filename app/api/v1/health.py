from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health check endpoint")
async def health_check():
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "llm_base_url_configured": bool(settings.LLM_BASE_URL),
        "llm_model": settings.LLM_MODEL_NAME,
    }

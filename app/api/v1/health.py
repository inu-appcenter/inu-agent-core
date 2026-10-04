import httpx
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health check endpoint")
async def health_check():
    portal_status = "unknown"
    portal_err = None
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.INU_PORTAL_SERVER_URL.rstrip('/')}/mcp")
            portal_status = f"HTTP {resp.status_code}"
    except Exception as e:
        portal_status = "failed"
        portal_err = str(e)

    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "llm_base_url_configured": bool(settings.LLM_BASE_URL),
        "llm_model": settings.LLM_MODEL_NAME,
        "portal_url": settings.INU_PORTAL_SERVER_URL,
        "portal_connectivity": portal_status,
        "portal_error": portal_err,
    }

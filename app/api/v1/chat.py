import json
from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse

from app.llm.schemas import ChatRequest
from app.orchestrator.engine import orchestrator
from app.core.logging import logger

router = APIRouter()


@router.post("/chat/stream", summary="Stream AI Agent chat response via SSE")
async def chat_stream(
    request: ChatRequest,
    req: Request,
    authorization: str = Header(default="", alias="Authorization"),
    x_appcenter_client: str = Header(default="", alias="X-AppCenter-Client"),
):
    # If header specifies client origin, override request client
    if x_appcenter_client:
        request.client = x_appcenter_client.upper()

    logger.info(f"Incoming chat request: '{request.message[:30]}...' from client: {request.client}")

    async def event_generator():
        try:
            async for event in orchestrator.run_stream(request):
                # Check for client disconnect
                if await req.is_disconnected():
                    logger.warning("Client disconnected from SSE stream")
                    break

                data_str = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False)
                yield f"data: {data_str}\n\n"
        except Exception as e:
            logger.error(f"Error in chat event generator: {e}", exc_info=True)
            err_data = json.dumps({"event_type": "ERROR", "error": str(e)}, ensure_ascii=False)
            yield f"data: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

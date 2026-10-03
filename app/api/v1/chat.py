import json
from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse

from app.llm.schemas import ChatRequest, ChatActionCallbackRequest
from app.orchestrator.engine import orchestrator
from app.core.logging import logger

router = APIRouter()


@router.post("/chat/stream", summary="Stream AI Agent chat response via SSE")
async def chat_stream(
    request: ChatRequest,
    req: Request,
    authorization: str = Header(default="", alias="Authorization"),
    auth: str = Header(default="", alias="Auth"),
    x_appcenter_client: str = Header(default="", alias="X-AppCenter-Client"),
):
    # If header specifies client origin, override request client
    if x_appcenter_client:
        request.client = x_appcenter_client.upper()

    raw_token = auth or authorization
    if not raw_token and request.client_context:
        raw_token = (
            request.client_context.get("accessToken")
            or request.client_context.get("access_token")
            or request.client_context.get("auth")
            or request.client_context.get("authorization")
            or request.client_context.get("token")
            or ""
        )
    if raw_token:
        if request.client_context is None:
            request.client_context = {}
        clean_token = str(raw_token).replace("Bearer ", "").strip()
        request.client_context["auth"] = clean_token
        request.client_context["authorization"] = f"Bearer {clean_token}"
        request.client_context["accessToken"] = clean_token

    logger.info(f"Incoming chat request: '{request.message[:30]}...' from client: {request.client} (Auth: {bool(raw_token)})")


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


@router.post("/chat/action-callback", summary="Resume AI Agent conversation stream via SSE using Client Action result")
async def chat_action_callback(
    callback_req: ChatActionCallbackRequest,
    req: Request,
    authorization: str = Header(default="", alias="Authorization"),
    auth: str = Header(default="", alias="Auth"),
    x_appcenter_client: str = Header(default="", alias="X-AppCenter-Client"),
):
    if x_appcenter_client:
        callback_req.client = x_appcenter_client.upper()

    raw_token = auth or authorization
    if not raw_token and callback_req.client_context:
        raw_token = (
            callback_req.client_context.get("accessToken")
            or callback_req.client_context.get("access_token")
            or callback_req.client_context.get("auth")
            or callback_req.client_context.get("authorization")
            or callback_req.client_context.get("token")
            or ""
        )
    if raw_token:
        if callback_req.client_context is None:
            callback_req.client_context = {}
        clean_token = str(raw_token).replace("Bearer ", "").strip()
        callback_req.client_context["auth"] = clean_token
        callback_req.client_context["authorization"] = f"Bearer {clean_token}"
        callback_req.client_context["accessToken"] = clean_token


    logger.info(
        f"Incoming action callback: action_id={callback_req.action_id} success={callback_req.success} "
        f"client={callback_req.client} session_id={callback_req.session_id}"
    )

    async def event_generator():
        try:
            async for event in orchestrator.resume_stream_with_action_result(callback_req):
                if await req.is_disconnected():
                    logger.warning("Client disconnected from action callback SSE stream")
                    break

                data_str = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False)
                yield f"data: {data_str}\n\n"
        except Exception as e:
            logger.error(f"Error in action callback event generator: {e}", exc_info=True)
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

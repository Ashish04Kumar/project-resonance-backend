"""HTTP API for conversation sessions and their local real-time voice workers."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import settings
from .conversation import call_store, generate_turn
from .conversation_pods import get_conversation_pod, pod_catalog
from .models import (
    CallStatusResponse,
    CreateCallRequest,
    CreateCallResponse,
    TurnRequest,
    TurnResponse,
    VoiceJoinResponse,
)
from .voice_agent import participant_token
from .voice_sessions import client_livekit_url, start_agent, stop_agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # This prototype intentionally keeps no sessions beyond process lifetime.


app = FastAPI(
    title="Project Synq Conversation API",
    version="0.1.0",
    description="Local development API for LangGraph conversation sessions and Pipecat voice calls.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "conversation-api"}


@app.get("/api/v1/conversation-pods")
async def list_conversation_pods() -> list[dict[str, object]]:
    """List available conversation capability bundles and their state."""
    return pod_catalog()


@app.post("/api/v1/calls", response_model=CreateCallResponse, status_code=status.HTTP_201_CREATED)
async def create_call(payload: CreateCallRequest) -> CreateCallResponse:
    if not payload.consent:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Explicit consent is required to begin.")
    try:
        pod = get_conversation_pod(settings.conversation_pod)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SYNQ_CONVERSATION_POD must be set to pod1 or pod2.",
        ) from exc
    session = await call_store.create(payload.language_mode, pod.id)
    use_gemini_live = settings.voice_provider == "gemini_live" and bool(settings.gemini_api_key)
    if use_gemini_live:
        greeting, provider = "", "gemini-live"
    else:
        greeting, provider = await generate_turn(session)
        if not provider.startswith(("gemini", "ollama")):
            call_store.remove(session.call_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neither local Ollama nor the configured Gemini text fallback is available. Check the backend configuration.",
            )
    try:
        livekit_url = client_livekit_url(payload.client_host)
        await start_agent(session, greeting)
    except (ValueError, RuntimeError) as exc:
        call_store.remove(session.call_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    logger.info("CALL API READY | call={} | provider={} | pod={}", session.call_id, provider, session.conversation_pod)
    return CreateCallResponse(
        call_id=session.call_id,
        conversation_pod=session.conversation_pod,
        assistant_text=greeting,
        language_mode=session.language_mode,
        elapsed_seconds=session.elapsed_seconds,
        phase=session.phase,
        provider=provider,
        livekit_url=livekit_url,
        livekit_token=participant_token(session.call_id, session.call_id),
    )


@app.post("/api/v1/calls/{call_id}/resume", response_model=VoiceJoinResponse)
async def resume_voice_call(call_id: str, payload: CreateCallRequest) -> VoiceJoinResponse:
    session = call_store.get(call_id)
    if session is None or session.phase == "closed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This call can no longer be resumed.")
    try:
        livekit_url = client_livekit_url(payload.client_host)
        if settings.voice_provider == "gemini_live" and settings.gemini_api_key:
            greeting = ""
        else:
            greeting, provider = await generate_turn(session)
            if not provider.startswith(("gemini", "ollama")):
                raise RuntimeError("Neither local Ollama nor the configured Gemini text fallback is available.")
        await start_agent(session, greeting)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return VoiceJoinResponse(
        call_id=call_id,
        assistant_text=greeting,
        livekit_url=livekit_url,
        livekit_token=participant_token(call_id, call_id),
    )


@app.post("/api/v1/calls/{call_id}/turn", response_model=TurnResponse)
async def post_turn(call_id: str, payload: TurnRequest) -> TurnResponse:
    session = call_store.get(call_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found or already deleted.")
    if session.phase == "closed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This call has ended.")
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A non-empty turn is required.")
    assistant, provider = await generate_turn(session, user_text)
    return TurnResponse(
        call_id=call_id,
        assistant_text=assistant,
        language_mode=session.language_mode,
        elapsed_seconds=session.elapsed_seconds,
        phase=session.phase,
        covered_goals=session.covered_goals,
        provider=provider,
    )


@app.get("/api/v1/calls/{call_id}", response_model=CallStatusResponse)
async def get_call(call_id: str) -> CallStatusResponse:
    session = call_store.get(call_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found or already deleted.")
    return CallStatusResponse(
        call_id=call_id,
        conversation_pod=session.conversation_pod,
        language_mode=session.language_mode,
        elapsed_seconds=session.elapsed_seconds,
        phase=session.phase,
        covered_goals=session.covered_goals,
    )


@app.post("/api/v1/calls/{call_id}/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_call(call_id: str) -> None:
    """End a call idempotently so racing/double end requests both succeed."""
    await stop_agent(call_id)
    session = call_store.remove(call_id)
    if session is not None:
        session.closed = True
        if settings.export_call_logs:
            try:
                await _export_call_log(session)
            except Exception:
                logger.exception("CALL LOG EXPORT FAILED | call={}", call_id)
        session.transcript.clear()
        session.export_transcript.clear()
        session.rolling_summary = ""
        logger.info("CALL ENDED | call={} | transcript cleared", call_id)
    else:
        logger.info("CALL END DUPLICATE | call={} | already ended; returning 204", call_id)


async def _export_call_log(session) -> None:
    """Write the opted-in final transcript and conductor state to a local JSON file."""
    if not settings.export_call_logs:
        return
    ist = ZoneInfo("Asia/Kolkata")
    ended_at = datetime.now(ist)
    filename = f"call_{ended_at.strftime('%Y-%m-%d_%H-%M-%S')}_IST_{session.call_id}.json"
    output_directory = settings.call_log_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / filename
    payload = {
        "call_id": session.call_id,
        "created_at_utc": session.created_at.isoformat(),
        "ended_at_ist": ended_at.isoformat(),
        "duration_seconds": session.elapsed_seconds,
        "conversation_pod": session.conversation_pod,
        "language_mode": session.language_mode,
        "provider": settings.voice_provider,
        "final_state": {
            "phase": "closed",
            "conversation": session.conversation_state.model_dump(mode="json"),
            "covered_goals": session.covered_goals,
            "rolling_summary": session.rolling_summary,
            "latest_user_text": session.latest_user_text,
            "recent_questions": session.recent_questions,
            "idle_nudge_details_used": session.idle_nudge_details_used,
        },
        "transcript": session.export_transcript or session.transcript,
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    await asyncio.to_thread(
        temporary_path.write_text,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    await asyncio.to_thread(temporary_path.replace, output_path)
    logger.info("CALL LOG EXPORTED | call={} | path={}", session.call_id, output_path)

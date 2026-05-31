
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import AskRequest
from app.services.streamer import stream_agent
from app.utils.logger import setup_logger
import asyncio
import json

router = APIRouter()
logger = setup_logger(__name__)


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    question = request.question.strip()
    logger.info(f"Stream request | question='{question}' | provider={request.provider}")

    async def event_generator():
        loop = asyncio.get_event_loop()
        q    = asyncio.Queue()

        def _produce():
            try:
                for chunk in stream_agent(question, provider=request.provider):
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    q.put(f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)
                
        loop.run_in_executor(None, _produce)

        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
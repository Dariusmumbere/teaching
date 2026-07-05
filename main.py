"""
Party Line — backend (single-file FastAPI app)

Setup
-----
1. pip install fastapi "uvicorn[standard]" sqlalchemy asyncpg pydantic python-multipart

2. Set environment variables before running:
     DATABASE_URL   -> your Postgres connection string (Neon, Render, etc.)
                        e.g. postgres://user:pass@host/dbname
     CORS_ORIGINS   -> (optional) comma-separated list of allowed frontend origins.
                        Defaults to "*" (any origin) if not set.

3. Run it:
     uvicorn main:app --host 0.0.0.0 --port 8000

   On Render, set the start command to:
     uvicorn main:app --host 0.0.0.0 --port $PORT

What it does
------------
Stores every question ("call") and its answer in Postgres. Audio is uploaded
as a file, stored as base64 text in the database, and served back the same
way the frontend already expects (a data: URL) — so no separate file storage
is needed to get this working. If you outgrow that (lots of long recordings),
swap the audio columns for a Backblaze B2 / S3 URL instead.
"""

import base64
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

RAW_DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not RAW_DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is required, e.g. "
        "postgres://user:password@host/dbname"
    )

DATABASE_URL = RAW_DATABASE_URL
# Normalize whatever scheme was given to the async driver SQLAlchemy needs.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# asyncpg wants SSL passed as a connect arg, not a "?sslmode=require" query string.
connect_args = {}
if "sslmode=require" in DATABASE_URL or "neon.tech" in DATABASE_URL:
    DATABASE_URL = re.sub(r"[?&]sslmode=require", "", DATABASE_URL)
    connect_args["ssl"] = True

engine = create_async_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, autoincrement=True)  # doubles as the "LINE 00N" number
    kind = Column(String(10), nullable=False)  # 'text' | 'audio'
    content = Column(Text, nullable=False)  # plain text, or a data: URL for audio
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    answer_kind = Column(String(10), nullable=True)
    answer_content = Column(Text, nullable=True)
    answer_created_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextIn(BaseModel):
    content: str


class CallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    content: str
    created_at: datetime
    answer_kind: Optional[str] = None
    answer_content: Optional[str] = None
    answer_created_at: Optional[datetime] = None


MAX_AUDIO_BYTES = 8 * 1024 * 1024  # 8MB per recording


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Party Line API")

_cors_env = os.environ.get("CORS_ORIGINS", "*").strip()
allow_origins = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/api/health")
async def health():
    return {"status": "the line is open"}


@app.get("/api/calls", response_model=List[CallOut])
async def list_calls():
    async with SessionLocal() as session:
        result = await session.execute(select(Call).order_by(Call.id.asc()))
        return result.scalars().all()


@app.post("/api/calls/text", response_model=CallOut)
async def create_text_call(payload: TextIn):
    content = payload.content.strip()
    if not content:
        raise HTTPException(400, "A question can't be empty.")
    async with SessionLocal() as session:
        call = Call(kind="text", content=content)
        session.add(call)
        await session.commit()
        await session.refresh(call)
        return call


@app.post("/api/calls/audio", response_model=CallOut)
async def create_audio_call(file: UploadFile = File(...)):
    data_url = await _read_audio_as_data_url(file)
    async with SessionLocal() as session:
        call = Call(kind="audio", content=data_url)
        session.add(call)
        await session.commit()
        await session.refresh(call)
        return call


@app.post("/api/calls/{call_id}/answer/text", response_model=CallOut)
async def answer_with_text(call_id: int, payload: TextIn):
    content = payload.content.strip()
    if not content:
        raise HTTPException(400, "An answer can't be empty.")
    async with SessionLocal() as session:
        call = await _get_open_call_or_error(session, call_id)
        call.answer_kind = "text"
        call.answer_content = content
        call.answer_created_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(call)
        return call


@app.post("/api/calls/{call_id}/answer/audio", response_model=CallOut)
async def answer_with_audio(call_id: int, file: UploadFile = File(...)):
    data_url = await _read_audio_as_data_url(file)
    async with SessionLocal() as session:
        call = await _get_open_call_or_error(session, call_id)
        call.answer_kind = "audio"
        call.answer_content = data_url
        call.answer_created_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(call)
        return call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_audio_as_data_url(file: UploadFile) -> str:
    data = await file.read()
    if not data:
        raise HTTPException(400, "That recording came through empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "That recording is too long (max ~8MB, about 90 seconds).")
    b64 = base64.b64encode(data).decode("ascii")
    mime = file.content_type or "audio/webm"
    return f"data:{mime};base64,{b64}"


async def _get_open_call_or_error(session: AsyncSession, call_id: int) -> Call:
    call = await session.get(Call, call_id)
    if not call:
        raise HTTPException(404, "That line doesn't exist.")
    if call.answer_content:
        raise HTTPException(409, "Someone already picked up this line.")
    return call

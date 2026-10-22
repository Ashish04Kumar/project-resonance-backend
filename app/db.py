"""Postgres connection pool and startup migrations.

`DATABASE_URL` in backend/.env points at the database. Migrations are plain
.sql files in app/migrations, applied in name order once each and recorded in
`schema_migrations`. If DATABASE_URL is not set, the API still starts (voice
calls keep working) and the account endpoints answer 503.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
from loguru import logger

from .config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_pool: asyncpg.Pool | None = None


async def connect() -> None:
    global _pool
    if not settings.database_url:
        logger.warning("DATABASE_URL is not set; account features are disabled")
        return
    _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
    await migrate(_pool)
    logger.info("DATABASE READY")


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool | None:
    return _pool


async def migrate(target: asyncpg.Pool) -> None:
    async with target.acquire() as connection:
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {row["name"] for row in await connection.fetch("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            async with connection.transaction():
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute("INSERT INTO schema_migrations (name) VALUES ($1)", path.name)
            logger.info("MIGRATION APPLIED | {}", path.name)

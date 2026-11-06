"""Engine and session factory. One engine (connection pool) per process."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(
        database_url,
        pool_pre_ping=True,  # drop dead connections (e.g. after a DB restart) transparently
        pool_size=10,
        max_overflow=5,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: objects stay readable after commit without another round trip.
    return async_sessionmaker(engine, expire_on_commit=False)

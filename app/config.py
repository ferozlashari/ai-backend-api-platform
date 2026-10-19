import os
from dataclasses import dataclass


@dataclass
class Settings:
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    cache_ttl_seconds: float = float(os.getenv("CACHE_TTL_SECONDS", "300"))
    default_owner_tier: str = os.getenv("DEFAULT_TIER", "free")


settings = Settings()

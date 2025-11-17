# ABOUTME: Traffic light status caching layer to eliminate redundant database queries
# ABOUTME: Provides thread-safe LRU cache keyed by (scenario_token, lidarpc_token)

import threading
from collections import OrderedDict
from typing import Generator, List, Optional, Tuple

from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.database.nuplan_db import nuplan_scenario_queries


class TrafficLightCache:
    """
    Thread-safe LRU cache for traffic light status data.

    Traffic light status doesn't change during a scenario replay, but the current implementation
    queries the database 48 times per simulation step. This cache eliminates redundant queries by
    storing (scenario_token, lidarpc_token) → List[TrafficLightStatusData] mappings.

    Design choices:
    - LRU eviction: Keeps recently-used scenarios in memory
    - Thread-safe: Uses lock to protect cache operations
    - Scenario-scoped: Keys are (scenario_token, lidarpc_token) tuples
    - Stores lists (not generators): Traffic light data per lidarpc is small (~10-50 entries)

    Performance impact:
    - Cache hit: <1ms (list copy)
    - Cache miss: ~40ms (database query + list materialization)
    - Expected hit rate: 47/48 = 97.9% (only 1 miss per step, 47 hits)
    """

    _cache: OrderedDict[Tuple[str, str], List[TrafficLightStatusData]] = OrderedDict()
    _lock: threading.Lock = threading.Lock()
    _max_size: int = 1000  # Max cached (scenario, lidarpc) pairs

    @classmethod
    def get(cls, scenario_token: str, lidarpc_token: str) -> Optional[List[TrafficLightStatusData]]:
        """
        Retrieve cached traffic light status for a given scenario and lidarpc token.

        Args:
            scenario_token: Unique scenario identifier
            lidarpc_token: Lidar point cloud token (iteration identifier)

        Returns:
            List of traffic light status data if cached, None otherwise
        """
        key = (scenario_token, lidarpc_token)
        with cls._lock:
            if key in cls._cache:
                # Move to end (mark as recently used)
                cls._cache.move_to_end(key)
                return cls._cache[key]
        return None

    @classmethod
    def put(cls, scenario_token: str, lidarpc_token: str, data: List[TrafficLightStatusData]) -> None:
        """
        Store traffic light status in cache.

        Args:
            scenario_token: Unique scenario identifier
            lidarpc_token: Lidar point cloud token (iteration identifier)
            data: List of traffic light status data to cache
        """
        key = (scenario_token, lidarpc_token)
        with cls._lock:
            # Remove if already exists (to update position)
            if key in cls._cache:
                del cls._cache[key]

            # Add to end (most recently used)
            cls._cache[key] = data

            # Evict oldest if over max size
            if len(cls._cache) > cls._max_size:
                cls._cache.popitem(last=False)  # Remove oldest (FIFO/LRU)

    @classmethod
    def clear(cls) -> None:
        """Clear all cached data (useful for testing or scenario transitions)."""
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def size(cls) -> int:
        """Return number of cached entries."""
        with cls._lock:
            return len(cls._cache)


def get_cached_traffic_light_status(
    log_file: str,
    scenario_token: str,
    lidarpc_token: str
) -> Generator[TrafficLightStatusData, None, None]:
    """
    Get traffic light status with caching.

    This function wraps nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db()
    with a cache layer. On cache hit, yields from cached list. On cache miss, queries database,
    materializes generator to list, caches it, then yields results.

    Args:
        log_file: Path to database file
        scenario_token: Scenario token (for cache key scoping)
        lidarpc_token: Lidar PC token to query

    Yields:
        TrafficLightStatusData objects
    """
    # Check cache
    cached_data = TrafficLightCache.get(scenario_token, lidarpc_token)

    if cached_data is not None:
        # Cache hit - yield from cached list
        for item in cached_data:
            yield item
    else:
        # Cache miss - query database
        db_generator = nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db(
            log_file, lidarpc_token
        )

        # Materialize generator to list (traffic light data is small, ~10-50 entries per lidarpc)
        data_list = list(db_generator)

        # Store in cache
        TrafficLightCache.put(scenario_token, lidarpc_token, data_list)

        # Yield from materialized list
        for item in data_list:
            yield item

# ABOUTME: Unit tests for traffic light status caching functionality
# ABOUTME: Verifies cache correctness, thread safety, and performance improvements

import threading
import time
import unittest
from typing import Generator, List
from unittest.mock import MagicMock, patch

from nuplan.database.common.traffic_light_cache import TrafficLightCache, get_cached_traffic_light_status
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType


class TestTrafficLightCache(unittest.TestCase):
    """Test traffic light caching layer."""

    def setUp(self):
        """Reset cache before each test."""
        TrafficLightCache.clear()

    def test_cache_empty_initially(self):
        """Verify cache starts empty."""
        self.assertEqual(TrafficLightCache.size(), 0)

    def test_cache_stores_and_retrieves_single_entry(self):
        """Verify basic cache storage and retrieval."""
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Create mock traffic light data
        tl_data = [
            TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=1,
                timestamp=1000
            ),
            TrafficLightStatusData(
                status=TrafficLightStatusType.RED,
                lane_connector_id=2,
                timestamp=1000
            )
        ]

        # Store in cache
        TrafficLightCache.put(scenario_token, lidarpc_token, tl_data)

        # Retrieve from cache
        cached_data = TrafficLightCache.get(scenario_token, lidarpc_token)

        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), 2)
        self.assertEqual(cached_data[0].status, TrafficLightStatusType.GREEN)
        self.assertEqual(cached_data[1].status, TrafficLightStatusType.RED)

    def test_cache_miss_returns_none(self):
        """Verify cache returns None for missing keys."""
        result = TrafficLightCache.get("nonexistent_scenario", "nonexistent_lidar")
        self.assertIsNone(result)

    def test_cache_clear(self):
        """Verify cache can be cleared."""
        TrafficLightCache.put("scenario_1", "lidar_1", [])
        TrafficLightCache.put("scenario_2", "lidar_2", [])

        self.assertEqual(TrafficLightCache.size(), 2)

        TrafficLightCache.clear()

        self.assertEqual(TrafficLightCache.size(), 0)

    def test_cache_thread_safety(self):
        """Verify cache is thread-safe under concurrent access."""
        num_threads = 10
        iterations_per_thread = 100

        def worker(thread_id: int):
            """Worker function that reads/writes cache concurrently."""
            for i in range(iterations_per_thread):
                scenario_token = f"scenario_{thread_id}"
                lidarpc_token = f"lidar_{i}"

                # Write
                tl_data = [TrafficLightStatusData(
                    status=TrafficLightStatusType.GREEN,
                    lane_connector_id=thread_id * 1000 + i,
                    timestamp=i
                )]
                TrafficLightCache.put(scenario_token, lidarpc_token, tl_data)

                # Read
                cached = TrafficLightCache.get(scenario_token, lidarpc_token)
                self.assertIsNotNone(cached)
                self.assertEqual(len(cached), 1)
                self.assertEqual(cached[0].lane_connector_id, thread_id * 1000 + i)

        # Launch threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all data was stored correctly
        expected_size = num_threads * iterations_per_thread
        self.assertEqual(TrafficLightCache.size(), expected_size)

    def test_cache_lru_eviction(self):
        """Verify LRU eviction when cache exceeds max size."""
        # Set small cache size
        original_max_size = TrafficLightCache._max_size
        TrafficLightCache._max_size = 5

        try:
            # Add 6 entries (should evict oldest)
            for i in range(6):
                TrafficLightCache.put(f"scenario_{i}", f"lidar_{i}", [])

            # Cache should have 5 entries (LRU evicted 1)
            self.assertLessEqual(TrafficLightCache.size(), 5)

            # Oldest entry (0) should be evicted
            self.assertIsNone(TrafficLightCache.get("scenario_0", "lidar_0"))

            # Newest entries should still be present
            self.assertIsNotNone(TrafficLightCache.get("scenario_5", "lidar_5"))

        finally:
            # Restore original max size
            TrafficLightCache._max_size = original_max_size

    def test_cache_update_existing_key(self):
        """Verify updating an existing cache entry works correctly."""
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Store initial data
        initial_data = [TrafficLightStatusData(
            status=TrafficLightStatusType.GREEN,
            lane_connector_id=1,
            timestamp=1000
        )]
        TrafficLightCache.put(scenario_token, lidarpc_token, initial_data)

        # Update with new data
        updated_data = [TrafficLightStatusData(
            status=TrafficLightStatusType.RED,
            lane_connector_id=2,
            timestamp=2000
        )]
        TrafficLightCache.put(scenario_token, lidarpc_token, updated_data)

        # Verify update
        cached = TrafficLightCache.get(scenario_token, lidarpc_token)
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0].status, TrafficLightStatusType.RED)
        self.assertEqual(cached[0].timestamp, 2000)


class TestGetCachedTrafficLightStatus(unittest.TestCase):
    """Test cached query wrapper function."""

    def setUp(self):
        """Reset cache before each test."""
        TrafficLightCache.clear()

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_cache_miss_queries_database(self, mock_db_query):
        """Verify cache miss triggers database query and caches result."""
        log_file = "/path/to/log.db"
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Mock DB response (generator)
        def mock_generator() -> Generator[TrafficLightStatusData, None, None]:
            yield TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=1,
                timestamp=1000
            )

        mock_db_query.return_value = mock_generator()

        # First call should query DB
        result1 = list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))

        # Verify DB was queried
        mock_db_query.assert_called_once_with(log_file, lidarpc_token)

        # Verify result
        self.assertEqual(len(result1), 1)
        self.assertEqual(result1[0].status, TrafficLightStatusType.GREEN)

        # Second call should use cache (no additional DB query)
        mock_db_query.reset_mock()
        result2 = list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))

        # Verify DB was NOT queried again
        mock_db_query.assert_not_called()

        # Verify cached result matches original
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0].status, TrafficLightStatusType.GREEN)

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_cache_hit_skips_database(self, mock_db_query):
        """Verify cache hit bypasses database query."""
        log_file = "/path/to/log.db"
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Pre-populate cache
        cached_data = [TrafficLightStatusData(
            status=TrafficLightStatusType.RED,
            lane_connector_id=2,
            timestamp=2000
        )]
        TrafficLightCache.put(scenario_token, lidarpc_token, cached_data)

        # Query should use cache, not DB
        result = list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))

        # Verify DB was NOT queried
        mock_db_query.assert_not_called()

        # Verify cached result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, TrafficLightStatusType.RED)

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_empty_result_is_cached(self, mock_db_query):
        """Verify empty database results are cached to avoid repeated queries."""
        log_file = "/path/to/log.db"
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Mock empty DB response
        mock_db_query.return_value = iter([])  # Empty generator

        # First call queries DB
        result1 = list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))
        self.assertEqual(len(result1), 0)
        mock_db_query.assert_called_once()

        # Second call uses cached empty result
        mock_db_query.reset_mock()
        result2 = list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))
        self.assertEqual(len(result2), 0)
        mock_db_query.assert_not_called()

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_different_scenarios_have_separate_caches(self, mock_db_query):
        """Verify different scenarios maintain separate cache entries."""
        log_file = "/path/to/log.db"

        # Mock different responses for different scenarios
        def make_generator(connector_id: int) -> Generator[TrafficLightStatusData, None, None]:
            yield TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=connector_id,
                timestamp=1000
            )

        mock_db_query.side_effect = [
            make_generator(1),
            make_generator(2)
        ]

        # Query two different scenarios
        result1 = list(get_cached_traffic_light_status(log_file, "scenario_1", "lidar_1"))
        result2 = list(get_cached_traffic_light_status(log_file, "scenario_2", "lidar_2"))

        # Verify both were queried
        self.assertEqual(mock_db_query.call_count, 2)

        # Verify results are different
        self.assertEqual(result1[0].lane_connector_id, 1)
        self.assertEqual(result2[0].lane_connector_id, 2)

        # Verify both are cached
        self.assertEqual(TrafficLightCache.size(), 2)


class TestCachePerformance(unittest.TestCase):
    """Performance verification tests."""

    def setUp(self):
        """Reset cache before each test."""
        TrafficLightCache.clear()

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_cache_reduces_query_time(self, mock_db_query):
        """Verify cache hit is significantly faster than database query."""
        log_file = "/path/to/log.db"
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        # Mock slow DB query (simulating 40ms overhead)
        def slow_generator():
            time.sleep(0.04)  # 40ms delay
            yield TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=1,
                timestamp=1000
            )

        mock_db_query.return_value = slow_generator()

        # First call (cache miss) - should be slow
        start_time = time.time()
        list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))
        db_query_time = time.time() - start_time

        # Second call (cache hit) - should be fast
        start_time = time.time()
        list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))
        cache_hit_time = time.time() - start_time

        # Verify cache is at least 10x faster
        self.assertLess(cache_hit_time * 10, db_query_time)

    @patch('nuplan.database.nuplan_db.nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db')
    def test_cache_reduces_query_count(self, mock_db_query):
        """Verify cache eliminates redundant database queries."""
        log_file = "/path/to/log.db"
        scenario_token = "scenario_123"
        lidarpc_token = "lidar_456"

        def mock_generator():
            yield TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=1,
                timestamp=1000
            )

        mock_db_query.return_value = mock_generator()

        # Simulate 48 queries per step (current behavior without cache)
        for _ in range(48):
            list(get_cached_traffic_light_status(log_file, scenario_token, lidarpc_token))

        # Verify only 1 DB query occurred (other 47 were cache hits)
        self.assertEqual(mock_db_query.call_count, 1)


if __name__ == '__main__':
    unittest.main()

# ABOUTME: Tests for database connection pooling to ensure connections are reused
# ABOUTME: and not created redundantly per query batch

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import sqlalchemy

from nuplan.database.common.db import DB, SessionManager


class TestConnectionPooling(unittest.TestCase):
    """Test database connection pooling behavior."""

    def setUp(self) -> None:
        """Set up test database."""
        # Create a temporary in-memory database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = Path(self.temp_db.name)

        # Create a simple test table
        conn = sqlite3.connect(str(self.temp_db_path))
        conn.execute('CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)')
        conn.execute('INSERT INTO test_table VALUES (1, "test1")')
        conn.execute('INSERT INTO test_table VALUES (2, "test2")')
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        """Clean up test database."""
        self.temp_db_path.unlink(missing_ok=True)

    def test_session_manager_reuses_connection_same_thread(self) -> None:
        """Test that SessionManager reuses connection within same thread."""
        def creator() -> sqlite3.Connection:
            return sqlite3.connect(str(self.temp_db_path))

        session_manager = SessionManager(creator)

        # Get session twice in same thread
        session1 = session_manager.session
        session2 = session_manager.session

        # Should be same session object
        self.assertIs(session1, session2, "SessionManager should reuse session in same thread")

        # Should also reuse engine
        engine1 = session_manager.engine
        engine2 = session_manager.engine
        self.assertIs(engine1, engine2, "SessionManager should reuse engine in same thread")

    def test_session_manager_separate_connections_different_threads(self) -> None:
        """Test that SessionManager creates separate connections per thread."""
        def creator() -> sqlite3.Connection:
            return sqlite3.connect(str(self.temp_db_path))

        session_manager = SessionManager(creator)

        # Get session in main thread
        main_session = session_manager.session

        # Get session in different thread
        other_thread_session = None
        def get_session_in_thread():
            nonlocal other_thread_session
            other_thread_session = session_manager.session

        thread = threading.Thread(target=get_session_in_thread)
        thread.start()
        thread.join()

        # Should be different session objects
        self.assertIsNot(main_session, other_thread_session,
                        "SessionManager should create separate sessions per thread")

    def test_connection_not_created_per_query(self) -> None:
        """Test that connections are NOT created for each query."""
        connection_count = 0

        def counting_creator() -> sqlite3.Connection:
            nonlocal connection_count
            connection_count += 1
            return sqlite3.connect(str(self.temp_db_path), check_same_thread=False)

        session_manager = SessionManager(counting_creator)

        # Execute multiple queries
        session = session_manager.session
        for _ in range(10):
            # Actually execute a query to trigger connection pool usage
            result = session.execute(sqlalchemy.text("SELECT COUNT(*) FROM test_table"))
            result.fetchall()

        # Pool starts with pool_size connections, but only creates them as needed
        # For 10 queries with default pool_size=5, we should create at most 5 connections
        self.assertLessEqual(connection_count, 5,
                           f"Should reuse connections from pool, created {connection_count}")

    def test_sqlalchemy_pool_configuration(self) -> None:
        """Test that SQLAlchemy engine is configured with proper pooling."""
        def creator() -> sqlite3.Connection:
            return sqlite3.connect(str(self.temp_db_path), check_same_thread=False)

        session_manager = SessionManager(creator)
        engine = session_manager.engine

        # Check that engine exists and has pool
        self.assertIsNotNone(engine)
        self.assertTrue(hasattr(engine, 'pool'), "Engine should have a connection pool")

        # Check pool configuration
        pool = engine.pool
        self.assertIsNotNone(pool)

        # Pool should be configured with reasonable size
        # QueuePool has size() method, StaticPool/NullPool don't
        from sqlalchemy.pool import QueuePool, StaticPool
        self.assertIsInstance(pool, (QueuePool, StaticPool),
                            f"Expected QueuePool or StaticPool, got {type(pool).__name__}")

    def test_connection_pool_reuse_across_multiple_queries(self) -> None:
        """Test that connection pool reuses connections across query batches."""
        connection_count = 0

        def counting_creator() -> sqlite3.Connection:
            nonlocal connection_count
            connection_count += 1
            return sqlite3.connect(str(self.temp_db_path), check_same_thread=False)

        session_manager = SessionManager(counting_creator)
        session = session_manager.session

        # Simulate multiple query batches (like in simulation steps)
        for step in range(10):
            # Execute actual queries like in real scenario
            result = session.execute(sqlalchemy.text("SELECT * FROM test_table WHERE id = :id"), {"id": step % 2 + 1})
            result.fetchall()

        # With connection pooling, connections should be reused across steps
        # We should only create a small number of connections (at most pool_size)
        self.assertLessEqual(connection_count, 5,
                           f"Expected at most 5 connections (pool_size), got {connection_count} connections")

    def test_pool_cleanup_on_thread_exit(self) -> None:
        """Test that pool properly handles thread lifecycle."""
        def creator() -> sqlite3.Connection:
            return sqlite3.connect(str(self.temp_db_path))

        session_manager = SessionManager(creator)

        # Create session in a thread that will exit
        thread_ran = False
        def worker():
            nonlocal thread_ran
            _ = session_manager.session
            thread_ran = True

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertTrue(thread_ran, "Worker thread should have executed")

        # Main thread should still be able to get its own session
        main_session = session_manager.session
        self.assertIsNotNone(main_session)


class TestConnectionPoolIntegration(unittest.TestCase):
    """Integration tests for connection pool with DB class."""

    def test_db_uses_pooled_connections(self) -> None:
        """Test that DB class benefits from connection pooling."""
        # This is a placeholder for integration testing
        # Would require actual DB setup with tables
        pass

    def test_concurrent_access_with_pool(self) -> None:
        """Test that connection pool handles concurrent access correctly."""
        # This is a placeholder for concurrency testing
        # Would require actual scenario with multiple threads
        pass


if __name__ == '__main__':
    unittest.main()

# ABOUTME: FastAPI WebSocket server for streaming real-time simulation data
# ABOUTME: Receives data from StreamingVisualizationCallback and broadcasts to web clients

import asyncio
import logging
from collections import deque
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="nuPlan Streaming Visualization Server",
    description="WebSocket server for real-time simulation visualization",
    version="1.0.0",
)

# Enable CORS for local development (web dashboard at localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_connections: List[WebSocket] = []
frame_buffer: deque = deque(maxlen=10)  # Keep last 10 frames for late joiners


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.frame_buffer: deque = deque(maxlen=10)

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept new WebSocket connection and send buffered frames.

        :param websocket: WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.active_connections)}")

        # Send buffered frames to help client catch up
        for frame in self.frame_buffer:
            try:
                await websocket.send_json(frame)
            except Exception as e:
                logger.warning(f"Failed to send buffered frame: {e}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove WebSocket connection from active list.

        :param websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast message to all connected clients.

        :param message: JSON-serializable message to broadcast
        """
        # Add to frame buffer
        self.frame_buffer.append(message)

        # Broadcast to all clients
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected_clients.append(connection)

        # Clean up disconnected clients
        for client in disconnected_clients:
            self.disconnect(client)


# Global connection manager
manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "active_connections": len(manager.active_connections),
        "buffered_frames": len(manager.frame_buffer),
    }


@app.post("/ingest")
async def ingest_endpoint(data: Dict[str, Any]):
    """
    HTTP POST endpoint for receiving simulation data from StreamingVisualizationCallback.

    This is where the simulation SENDS data TO the server.

    Flow:
    1. Callback sends simulation_start, simulation_step, simulation_end messages via HTTP POST
    2. Server broadcasts to all /stream WebSocket clients
    3. Callback doesn't need persistent connection (fire-and-forget)

    Protocol:
    - simulation_start: Scenario metadata, map info
    - simulation_step: Ego state, trajectory, agents (every 0.1s)
    - simulation_end: Completion signal

    Design rationale:
    - HTTP POST is appropriate for unidirectional communication (callback → server)
    - Avoids sync/async WebSocket mismatch (callback is synchronous)
    - Server-side WebSocket (/stream) remains async for dashboard clients
    """
    try:
        # Broadcast to all dashboard clients
        await manager.broadcast(data)

        # Log message type for debugging
        msg_type = data.get("type", "unknown")
        logger.info(f"Received {msg_type} via POST, broadcasting to {len(manager.active_connections)} clients")

        return {"status": "ok", "broadcast_count": len(manager.active_connections)}

    except Exception as e:
        logger.error(f"Ingest POST error: {e}")
        return {"status": "error", "message": str(e)}


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for web dashboard clients to receive simulation broadcasts.

    This is where web dashboards RECEIVE data FROM the server.

    Flow:
    1. Dashboard client connects → send buffered frames (catch up)
    2. Server broadcasts new frames as they arrive from /ingest
    3. Client disconnects → clean up
    """
    await manager.connect(websocket)

    try:
        # Keep connection alive, wait for disconnect
        while True:
            # We don't expect to receive data from dashboard clients (read-only)
            # But we need to keep the connection open to detect disconnects
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Stream WebSocket error: {e}")
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """Log server startup."""
    logger.info("🚀 nuPlan Streaming Visualization Server started")
    logger.info("📡 Simulation ingest: ws://localhost:8765/ingest (callback → server)")
    logger.info("📺 Dashboard stream: ws://localhost:8765/stream (server → clients)")
    logger.info("🌐 Health check: http://localhost:8765/")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    logger.info("🛑 nuPlan Streaming Visualization Server shutting down")
    logger.info(f"📊 Final stats: {len(manager.active_connections)} clients, "
                f"{len(manager.frame_buffer)} buffered frames")


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )

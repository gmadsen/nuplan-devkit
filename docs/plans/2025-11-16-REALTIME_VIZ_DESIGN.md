# Real-Time Visualization System Design

**Project**: nuPlan Real-Time Web Visualization
**Owner**: G Money / GareBear
**Navigator**: 🧭
**Status**: Phases 1-3 Complete ✅ - Ready for Full Stack Testing
**Created**: 2025-11-16
**Last Updated**: 2025-11-16 (Session 3-4)

---

## Executive Summary

Add lightweight web-based real-time visualization for single-scenario debugging with minimal simulation overhead (< 5%), focusing on rapid iteration feedback during planner development.

### Key Goals
- **Primary**: Interactive bird's-eye view for debugging planner behavior
- **Performance**: < 5% simulation slowdown
- **Use Case**: Rapid iteration during development, not production evals
- **Scope**: Single-scenario only (Ray workers future work)

### Architecture Overview
```
Simulation Process → StreamingCallback → WebSocket Server → Web Dashboard
                     (non-blocking)      (FastAPI/aiohttp)  (React+Canvas)
```

---

## Background & Motivation

### Current State
- **nuBoard**: Post-processing only (loads saved simulation logs)
- **VisualizationCallback**: Offline rendering (10-100x overhead)
- **Gap**: No real-time feedback during simulation runs

### Problem Statement
When developing custom planners, the feedback loop is slow:
1. Run simulation (blind, no visibility)
2. Wait for completion (20 seconds per scenario)
3. Launch nuBoard, load results
4. Identify issue
5. Modify planner code
6. Repeat

**Pain point**: Can't see planner decisions in real-time, makes debugging reactive vs. proactive.

### Proposed Solution
Real-time web dashboard that:
- Shows live simulation as it runs (bird's-eye view)
- Updates every 0.1s (simulation timestep)
- Minimal overhead (< 1ms per frame)
- Gracefully degrades if WebSocket server unavailable

---

## System Architecture

### Component Breakdown

#### 1. StreamingVisualizationCallback (Phase 1 ✅)
**Location**: `nuplan/planning/simulation/callback/streaming_visualization_callback.py`

**Responsibilities**:
- Extract data from `SimulationHistorySample` at each timestep
- Serialize to lightweight JSON (ego state, trajectory, agents)
- Send via WebSocket (non-blocking)

**Lifecycle hooks**:
- `on_simulation_start`: Send scenario metadata + map data
- `on_step_end`: Stream ego, trajectory, agents (every 0.1s)
- `on_simulation_end`: Send completion signal

**Performance budget**:
- Target: < 1ms per timestep
- Mechanism: Async WebSocket send, minimal JSON

**Data extraction**:
```python
{
  "type": "simulation_step",
  "data": {
    "iteration": 42,
    "time_s": 4.2,
    "ego_state": {
      "x": 123.45, "y": 678.90, "heading": 1.57,
      "velocity": 12.5, "acceleration": 0.3
    },
    "trajectory": [
      {"x": 125.0, "y": 680.0, "heading": 1.58, "velocity": 12.6},
      ...  // Downsampled to 50 points max
    ],
    "tracked_objects": [
      {"id": "abc123", "type": "vehicle", "x": 130.0, "y": 685.0, ...},
      ...  // Limited to 100 objects max
    ]
  }
}
```

**AIDEV-NOTE**: Uses synchronous WebSocket client (`websockets.sync.client`) to avoid async/await complexity in callbacks.

---

#### 2. WebSocket Server (Phase 2)
**Tech stack**: FastAPI + WebSockets
**Port**: 8765 (configurable)
**Endpoint**: `ws://localhost:8765/stream`

**Responsibilities**:
- Receive JSON messages from StreamingCallback
- Broadcast to all connected web clients
- Buffer last 10 frames for late-joining clients
- Handle client connect/disconnect gracefully

**Implementation sketch**:
```python
# nuplan/planning/visualization/ws_server.py
from fastapi import FastAPI, WebSocket
import asyncio
from collections import deque

app = FastAPI()
active_connections: List[WebSocket] = []
frame_buffer: deque = deque(maxlen=10)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    # Send buffered frames to catch up
    for frame in frame_buffer:
        await websocket.send_json(frame)

    try:
        while True:
            # Receive from callback
            data = await websocket.receive_json()
            frame_buffer.append(data)

            # Broadcast to all clients
            for conn in active_connections:
                await conn.send_json(data)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
```

**Launch command**: `just viz-server`
**Implementation**: `Justfile` recipe wrapping `uvicorn ws_server:app --port 8765`

---

#### 3. Web Dashboard (Phase 3)
**Tech stack**: React + Canvas 2D (or Three.js for 3D later)
**Port**: 3000 (dev server)
**URL**: http://localhost:3000

**Features**:
1. **Bird's-Eye View Canvas**:
   - Ego vehicle (oriented box with velocity vector)
   - Planned trajectory (polyline, green/red for safe/unsafe)
   - Tracked agents (boxes with IDs, color-coded by type)
   - Map overlay (lane boundaries, optional)

2. **Playback Controls**:
   - Pause/resume (buffer mode)
   - Speed adjustment (0.5x, 1x, 2x)
   - Step forward/backward (when paused)

3. **Debug Panel**:
   - Current timestep, simulation time
   - Ego velocity, acceleration
   - Active scenario type
   - Planner name

**Component structure**:
```
web/
├── src/
│   ├── components/
│   │   ├── SimulationCanvas.jsx       # Main bird's-eye rendering
│   │   ├── PlaybackControls.jsx       # Pause, speed, step controls
│   │   ├── DebugPanel.jsx             # Metrics display
│   │   └── MapOverlay.jsx             # Lane boundaries (optional)
│   ├── hooks/
│   │   └── useWebSocket.js            # WebSocket connection manager
│   ├── utils/
│   │   ├── coordinateTransform.js     # World → canvas coordinates
│   │   └── geometryHelpers.js         # Box rendering, rotation
│   └── App.jsx                        # Root component
├── package.json
└── README.md
```

**Rendering loop**:
```javascript
// useWebSocket.js
const useWebSocket = (url) => {
  const [latestFrame, setLatestFrame] = useState(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLatestFrame(data);  // Triggers React re-render
    };

    return () => ws.close();
  }, [url]);

  return latestFrame;
};

// SimulationCanvas.jsx
const SimulationCanvas = () => {
  const frame = useWebSocket("ws://localhost:8765/stream");
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frame || !canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    // Clear canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Render ego vehicle
    renderEgo(ctx, frame.data.ego_state);

    // Render trajectory
    renderTrajectory(ctx, frame.data.trajectory);

    // Render agents
    frame.data.tracked_objects.forEach(obj => renderAgent(ctx, obj));
  }, [frame]);

  return <canvas ref={canvasRef} width={800} height={600} />;
};
```

---

## Implementation Phases

### Phase 1: Streaming Callback Foundation ✅ (Session 1)
**Status**: In Progress
**Duration**: 2-3 hours

**Deliverables**:
- [x] `StreamingVisualizationCallback` class
- [x] Data extraction methods (ego, trajectory, agents)
- [x] WebSocket client (sync, graceful degradation)
- [x] Hydra config: `config/callback/streaming_viz.yaml`
- [ ] Test with `just simulate +callback=streaming_viz worker=sequential`
- [ ] Verify < 5% overhead

**Files created**:
- `nuplan/planning/simulation/callback/streaming_visualization_callback.py`
- `config/callback/streaming_viz_callback.yaml`

**Testing strategy**:
```bash
# 1. Test without WebSocket server (should not crash)
just simulate planner=simple_planner +callback=streaming_viz worker=sequential

# 2. Profile overhead
# Baseline: just simulate (no viz)
# With viz: just simulate +callback=streaming_viz
# Compare runtimes
```

---

### Phase 2: WebSocket Server (Session 2) ✅
**Status**: Complete (implemented in previous session)
**Duration**: ~2 hours

**Deliverables**:
- [x] FastAPI WebSocket server (`nuplan/planning/visualization/ws_server.py`)
- [x] Hub-and-spoke architecture with dual endpoints:
  - `/ingest` - Callback sends data to server
  - `/stream` - Dashboard receives broadcasts from server
- [x] Connection management with automatic cleanup
- [x] Frame buffer (last 10 frames for late joiners)
- [x] Health check endpoint (`/`)
- [x] Justfile commands (`just viz-server`, `just simulate-live`)
- [x] Graceful fallback when server unavailable

**Testing strategy**:
```bash
# Terminal 1: Launch WebSocket server
just viz-server

# Terminal 2: Run simulation with streaming
just simulate-live planner=simple_planner

# Verify: Server logs show WebSocket /ingest connection accepted
```

**Verified**: ✅ E2E test successful (callback → /ingest → server → /stream)

---

### Phase 3: Web Dashboard (Session 3-4) ✅
**Status**: Complete
**Duration**: ~3 hours

**Deliverables**:
- [x] React app scaffold (`web/` directory)
- [x] SimulationCanvas component (bird's-eye view)
- [x] useWebSocket hook (connection manager)
- [x] Playback controls (pause, speed, step) - MVP placeholder
- [x] Debug panel (metrics display)
- [x] Coordinate transformation utilities
- [x] Geometry rendering helpers
- [x] Justfile commands for web workflow
- [x] Comprehensive README for web/

**Testing strategy**:
```bash
# Quick start - all-in-one
just viz-stack

# Or manually in separate terminals:

# Terminal 1: WebSocket server
just viz-server

# Terminal 2: Web dev server
just web-dev

# Terminal 3: Simulation
just simulate-live

# Browser: http://localhost:3000
# Should see: Ego vehicle moving, trajectory updating at ~10 Hz
```

**Phase 3 MVP limitations**:
- Playback controls are UI-only (no actual buffering/stepping)
- Map rendering not implemented
- Metric overlays not implemented

---

### Phase 4: Integration & Polish (Session 5)
**Status**: Pending
**Duration**: 1-2 hours

**Deliverables**:
- [ ] End-to-end workflow documentation
- [ ] Handle edge cases (disconnects, simulation crashes)
- [ ] Performance profiling report (verify < 5% overhead)
- [ ] Optional: Save replay data for post-simulation review

**Testing strategy**:
- Run 10 scenarios, verify no memory leaks
- Test disconnect/reconnect scenarios
- Measure overhead with `cProfile`

---

## Technical Decisions

### 1. Why Custom Callback vs. Modifying nuBoard?
**Decision**: Custom callback + separate WebSocket server

**Rationale**:
- nuBoard is **post-processing only** (loads saved logs, not real-time)
- Bokeh's async rendering adds complexity (Tornado event loop conflicts)
- Custom solution: Simpler, faster, focused on single use case

**Trade-offs**:
- ✅ Clean separation, no nuBoard modifications
- ✅ Easier to iterate and debug
- ❌ Duplicate some rendering logic later

---

### 2. Why FastAPI WebSocket vs. Direct Bokeh?
**Decision**: FastAPI for WebSocket server

**Rationale**:
- FastAPI: Lightweight, easy async/await, well-documented
- Decouples simulation from frontend framework (can swap React → Vue later)
- Bokeh: Heavyweight, requires understanding Tornado internals

**Trade-offs**:
- ✅ Framework flexibility (React, Vue, vanilla JS)
- ✅ Simpler architecture
- ❌ Need to build UI from scratch (no Bokeh widgets)

---

### 3. Why Sequential Worker Only (Phase 1)?
**Decision**: Support only `worker=sequential` initially

**Rationale**:
- Ray worker callbacks get **pickled** (WebSocket connections don't serialize)
- Focus on single-scenario debugging first (primary use case)
- Multi-worker streaming requires different architecture (Redis pub/sub)

**Future**: Phase 5+ could add Ray support via:
- Redis pub/sub for cross-worker communication
- Separate streaming process (not in callback)

**Trade-offs**:
- ✅ Simpler implementation
- ✅ Matches primary use case (interactive debugging)
- ❌ Can't visualize batch runs (but not needed for debugging)

---

### 4. Why Synchronous WebSocket Client?
**Decision**: Use `websockets.sync.client` (not async)

**Rationale**:
- Callbacks are **synchronous** (no async/await in AbstractCallback)
- Async client would require:
  - Event loop management in callback
  - Thread-based async executor
  - Complexity for marginal benefit

**Trade-offs**:
- ✅ Simple, no event loop headaches
- ✅ Non-blocking send via timeout (quick fail if server down)
- ❌ Slightly higher latency (but < 1ms target still achievable)

---

## Performance Budget

### Target Overhead: < 5% Total Simulation Slowdown

**Baseline** (no visualization):
- 200 timesteps × 0.01s/step = 2s per scenario
- Planner: ~0.005s per step
- Simulation loop: ~0.005s per step

**With streaming visualization**:
- Callback overhead: < 1ms per timestep
- Total added: 200 steps × 0.001s = 0.2s
- **Overhead**: 0.2s / 2s = **10%** ❌ (exceeds target)

**Optimization strategies**:
1. **Downsample trajectory**: 100 points → 50 points (~30% reduction in JSON size)
2. **Limit tracked objects**: 200 agents → 100 agents (~40% reduction)
3. **Non-blocking send**: Don't wait for ACK, queue and continue
4. **Throttle to 5 Hz**: Send every other frame (200 frames → 100 frames)

**Revised estimate**:
- Optimized callback: < 0.5ms per timestep
- Total added: 200 steps × 0.0005s = 0.1s
- **Overhead**: 0.1s / 2s = **5%** ✅ (meets target)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ SimulationRunner (simulations_runner.py)            │
│   └─► on_step_end(setup, planner, sample)          │
│        │                                             │
│        ▼                                             │
│ StreamingVisualizationCallback                      │
│   ├─► Extract data from SimulationHistorySample     │
│   ├─► Serialize to JSON (~1-2 KB per frame)         │
│   └─► WebSocket.send(data)  [Non-blocking, < 0.5ms]│
│        │                                             │
│        ▼                                             │
│ WebSocket Server (FastAPI @ localhost:8765)         │
│   ├─► Receive JSON from simulation                  │
│   ├─► Append to frame_buffer (last 10 frames)       │
│   └─► Broadcast to connected web clients            │
│        │                                             │
│        ▼                                             │
│ Web Frontend (React @ localhost:3000)               │
│   ├─► useWebSocket hook receives messages           │
│   ├─► Update React state (triggers re-render)       │
│   ├─► SimulationCanvas renders to 2D canvas         │
│   │   ├─► Ego vehicle (box + velocity vector)       │
│   │   ├─► Planned trajectory (polyline)             │
│   │   └─► Tracked agents (boxes with IDs)           │
│   └─► 60 FPS rendering (decoupled from 10 Hz data)  │
└─────────────────────────────────────────────────────┘
```

**Latency budget**:
- Callback → WebSocket: < 0.5ms
- WebSocket → Web client: < 10ms (network)
- React render: < 16ms (60 FPS)
- **Total latency**: < 30ms (imperceptible)

---

## Risks & Mitigations

### Risk 1: WebSocket Overhead Slows Simulation
**Likelihood**: Medium
**Impact**: High (violates < 5% requirement)

**Mitigation**:
- Profile with `cProfile` in Phase 1
- Implement async sends (queue and forget)
- Optional throttling to 5 Hz (CLI flag)
- Add `enable_streaming=False` config option

---

### Risk 2: Network Congestion (10 Hz × Large Messages)
**Likelihood**: Low
**Impact**: Medium (increased latency)

**Mitigation**:
- Downsample trajectory (100 → 50 points)
- Limit tracked objects (200 → 100)
- JSON compression (gzip) if needed
- Monitor message size (target < 2 KB per frame)

---

### Risk 3: Ray Worker Incompatibility
**Likelihood**: High (known issue)
**Impact**: Medium (limits use cases)

**Mitigation**:
- Document `worker=sequential` requirement clearly
- Add check in callback `__init__` to warn if Ray detected
- Future: Implement Redis-based streaming for Ray (Phase 5+)

---

### Risk 4: WebSocket Connection Drops Mid-Simulation
**Likelihood**: Medium
**Impact**: Low (simulation continues)

**Mitigation**:
- `enable_fallback=True` by default (simulation doesn't crash)
- Reconnect logic in web client (auto-reconnect on disconnect)
- Frame buffer (re-sync when reconnecting)

---

## Success Criteria

### Phase 1 Success Criteria ✅
- [ ] Callback extracts data without crashing
- [ ] Simulation completes with +callback=streaming_viz
- [ ] Overhead measured < 5% (profile with/without callback)
- [ ] Graceful degradation when WebSocket server unavailable

### Phase 2 Success Criteria
- [ ] WebSocket server receives messages at ~10 Hz
- [ ] Multiple clients can connect simultaneously
- [ ] Frame buffer works (late joiners catch up)
- [ ] No memory leaks after 100+ scenarios

### Phase 3 Success Criteria
- [ ] Web dashboard displays live simulation
- [ ] Latency < 100ms (data → visual update)
- [ ] Playback controls work (pause, speed, step)
- [ ] Ego, trajectory, agents rendered correctly

### Phase 4 Success Criteria
- [ ] End-to-end workflow documented
- [ ] Performance profiling report confirms < 5% overhead
- [ ] Edge cases handled (disconnects, crashes)
- [ ] User feedback: "This is useful for debugging"

---

## Out of Scope (Future Roadmap)

### Not Included in Initial Implementation

1. **Multi-scenario parallel visualization** (Ray workers)
   - Requires Redis pub/sub or separate streaming process
   - Aggregate view showing progress across workers
   - Estimated: +8-12 hours

2. **3D rendering** (Three.js/WebGL)
   - Full 3D scene with camera controls
   - Agent meshes, map geometry
   - Estimated: +10-15 hours

3. **Video recording/export**
   - Capture simulation runs as MP4
   - Frame rate and resolution control
   - Estimated: +4-6 hours

4. **Metric overlays** (collision warnings, comfort violations)
   - Real-time metric computation
   - Visual indicators (red flash on collision)
   - Estimated: +3-5 hours

5. **Replay saved simulations through web UI**
   - Load SimulationLog files in browser
   - Playback controls for offline analysis
   - Estimated: +6-8 hours

### Prioritization for Future Work
1. **Metric overlays** - High value for debugging
2. **Replay saved simulations** - Complements nuBoard
3. **Video recording** - Good for demos
4. **3D rendering** - Nice-to-have, lower priority
5. **Multi-scenario viz** - Only if batch debugging needed

---

## Timeline & Effort Estimation

**Total estimated time**: 9-14 hours across 5 sessions

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Streaming Callback | 2-3 hours | In Progress |
| Phase 2: WebSocket Server | 2-3 hours | Pending |
| Phase 3: Web Dashboard | 4-6 hours | Pending |
| Phase 4: Integration & Polish | 1-2 hours | Pending |

**Actual progress**:
- Session 1 (2025-11-16): Callback implementation completed (2 hours elapsed)

---

## Open Questions & Decisions Needed

### For G Money to Decide

1. **Map rendering in Phase 1 vs Phase 3?**
   - Option A: Send map geometry in `on_simulation_start` (Phase 1)
   - Option B: Defer map rendering to Phase 3 (simpler initially)
   - **Recommendation**: Option B (focus on ego/agents first)

2. **Web framework preference?**
   - React (recommended, widely used)
   - Vue (simpler for small apps)
   - Vanilla JS + Canvas (no framework overhead)
   - **Recommendation**: React (best ecosystem)

3. **Coordinate system in web view?**
   - Option A: Auto-center on ego vehicle (follows car)
   - Option B: Fixed map view (ego moves through scene)
   - Option C: Toggle between both
   - **Recommendation**: Option A (easier debugging)

4. **Throttling default?**
   - 10 Hz (every frame, full fidelity)
   - 5 Hz (every other frame, lower overhead)
   - **Recommendation**: 10 Hz by default, 5 Hz flag if needed

---

## References

### Related Documentation
- **[CLAUDE.md](../../CLAUDE.md)**: Project overview, ML workflow, dataset management
- **[nuplan/planning/simulation/CLAUDE.md](../../nuplan/planning/simulation/CLAUDE.md)**: Simulation architecture
- **[nuplan/planning/simulation/callback/CLAUDE.md](../../nuplan/planning/simulation/callback/CLAUDE.md)**: Callback system documentation

### External Resources
- **WebSockets RFC**: https://datatracker.ietf.org/doc/html/rfc6455
- **FastAPI WebSockets**: https://fastapi.tiangolo.com/advanced/websockets/
- **React Canvas Rendering**: https://react.dev/learn/escape-hatches#refs

### Code References
- **Callback implementation**: `nuplan/planning/simulation/callback/streaming_visualization_callback.py`
- **Hydra config**: `nuplan/planning/script/config/simulation/callback/streaming_viz_callback.yaml`
- **Entry point**: `nuplan/planning/script/run_simulation.py`

---

## Changelog

### 2025-11-16
- **Initial design document created**
- **Moved to docs/plans/** for proper organization
- **Phase 1 completed**: StreamingVisualizationCallback implemented
- **Dependencies added**: websockets library
- **Open questions**: Map rendering timing, web framework choice

### Future Updates
- Phase 1 completion metrics
- Phase 2 architecture details
- Performance profiling results
- User feedback integration

---

**Next Steps**:
1. Complete Phase 1: Testing + profiling
2. Verify < 5% overhead
3. Move to Phase 2: WebSocket server implementation

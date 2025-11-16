# Phase 3 Web Dashboard Testing Guide

## Quick Test (Phase 2 incomplete - stub only)

**Current Status**: Phase 2 (WebSocket server) is not yet implemented, so end-to-end testing is blocked.

**What you can test now**:
1. ✅ Web dashboard UI loads correctly
2. ✅ WebSocket connection attempt (will fail gracefully)
3. ✅ UI components render properly
4. ❌ Live data streaming (requires Phase 2 server)

## Testing the UI in Standalone Mode

```bash
# 1. Start web dev server
just web-dev

# 2. Open browser to http://localhost:3000
# You should see:
#   - Header with "nuPlan Real-Time Visualization"
#   - Connection indicator showing "Disconnected" (red)
#   - Empty canvas (waiting for data)
#   - Debug panel showing "Waiting for simulation data..."
```

**Expected behavior**:
- Dashboard loads without errors
- Connection indicator shows "Disconnected" → "Connecting..." → "Disconnected" (retry loop)
- No console errors except WebSocket connection failures
- Reconnect button is functional

## Manual Component Testing

You can manually test components by modifying `App.jsx` to inject mock data:

```javascript
// In App.jsx, replace useWebSocket with mock data:
const mockFrame = {
  type: 'simulation_step',
  data: {
    iteration: 42,
    time_s: 4.2,
    ego_state: {
      x: 100.0,
      y: 200.0,
      heading: 1.57,
      velocity: 12.5,
      acceleration: 0.3,
      steering_angle: 0.05,
    },
    trajectory: [
      { x: 105.0, y: 205.0, heading: 1.58, velocity: 12.6 },
      { x: 110.0, y: 210.0, heading: 1.59, velocity: 12.7 },
    ],
    tracked_objects: [
      {
        id: 'vehicle_001',
        type: 'vehicle',
        x: 120.0,
        y: 220.0,
        heading: 1.6,
        velocity_x: 10.0,
        velocity_y: 0.5,
        length: 4.5,
        width: 2.0,
      },
    ],
  },
}

// Then replace:
const { latestFrame, ... } = useWebSocket(...)
// With:
const latestFrame = mockFrame
const connectionState = 'connected'
const isConnected = true
const reconnect = () => console.log('Reconnect')
```

## Full End-to-End Test (Phase 2 Required)

**⚠️ This requires Phase 2 WebSocket server to be implemented first!**

When Phase 2 is complete, test with:

```bash
# Terminal 1: Start WebSocket server
just viz-server

# Terminal 2: Start web dashboard
just web-dev

# Terminal 3: Run simulation with streaming
just simulate-live

# Browser: http://localhost:3000
# Should see:
#   - Connection indicator: green "Connected"
#   - Ego vehicle rendering (green box)
#   - Trajectory (cyan dashed line)
#   - Tracked agents (colored boxes)
#   - Debug panel updating at ~10 Hz
```

**Or use the all-in-one command**:
```bash
just viz-test
```

## Validation Checklist

### UI Validation
- [ ] Dashboard loads without errors
- [ ] Header shows correct title
- [ ] Connection indicator changes color based on state
- [ ] Reconnect button is clickable
- [ ] Debug panel displays correct sections
- [ ] Canvas renders with correct dimensions
- [ ] Zoom controls are functional

### WebSocket Connection (Phase 2)
- [ ] Connection indicator shows "Connecting..." on startup
- [ ] Connection indicator shows "Connected" when server available
- [ ] Auto-reconnect works after server restart
- [ ] Reconnect button manually triggers connection attempt
- [ ] Connection errors logged to console

### Rendering (Phase 2)
- [ ] Ego vehicle renders as green box
- [ ] Velocity vector shows correct direction
- [ ] Trajectory renders as cyan dashed line
- [ ] Tracked agents render with correct colors
- [ ] Canvas updates at ~10 Hz
- [ ] Zoom controls adjust scale correctly
- [ ] Grid overlay renders correctly

### Debug Panel (Phase 2)
- [ ] Scenario info displays on simulation_start
- [ ] Ego state updates every frame
- [ ] Metrics show correct units (m/s, m/s², degrees)
- [ ] High velocity/acceleration highlighted in red
- [ ] Trajectory point count matches
- [ ] Tracked object count matches

### Performance (Phase 2)
- [ ] Rendering latency < 100ms
- [ ] No dropped frames during simulation
- [ ] Memory usage stable over 10+ scenarios
- [ ] No console errors during normal operation

## Known Issues (Phase 3 MVP)

### Functional Limitations
1. **Playback controls**: UI-only, not functional
   - Pause button doesn't actually pause stream
   - Speed control doesn't affect rendering
   - No frame stepping capability

2. **Map rendering**: Not implemented
   - No lane boundaries
   - No crosswalks
   - Only grid overlay for reference

3. **Metric overlays**: Not implemented
   - No collision warnings
   - No comfort violation indicators
   - No visual alerts

### Expected Warnings
- "WebSocket connection failed" when server not running (expected)
- npm audit warnings about vulnerabilities (low severity, dev dependencies)

## Troubleshooting

### Web dashboard won't start
```bash
# Reinstall dependencies
cd web
rm -rf node_modules package-lock.json
npm install
```

### WebSocket connection fails
- Ensure Phase 2 server is implemented and running
- Check server logs for errors
- Verify port 8765 is not in use: `lsof -i :8765`

### Canvas not rendering
- Check browser console for errors
- Verify mock data structure matches protocol
- Test with browser DevTools canvas inspection

### Performance issues
- Reduce zoom level (increase metersPerPixel)
- Check CPU usage in browser DevTools
- Verify GPU acceleration is enabled

## Next Steps (Phase 4)

When ready for Phase 4 (Integration & Polish):
1. Test with Phase 2 WebSocket server
2. Profile end-to-end latency
3. Verify < 5% simulation overhead
4. Test with multiple scenarios
5. Document edge cases and limitations

---

**Status**: Phase 3 Complete, Phase 2 blocking full E2E tests
**Blocked By**: WebSocket server implementation (Phase 2)
**Ready For**: UI testing, component validation, mock data testing

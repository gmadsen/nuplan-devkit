// ABOUTME: Main application component for nuPlan real-time visualization dashboard
// ABOUTME: Integrates WebSocket connection, canvas rendering, and debug panels

import React, { useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { SimulationCanvas } from './components/SimulationCanvas'
import { DebugPanel } from './components/DebugPanel'
import { PlaybackControls } from './components/PlaybackControls'

/**
 * App - Main application root component.
 *
 * Layout:
 * ┌────────────────────────────────────────────────────┐
 * │ Header                                             │
 * ├────────────────────────────────┬───────────────────┤
 * │                                │                   │
 * │  SimulationCanvas              │  DebugPanel       │
 * │  (bird's-eye view)             │  (metrics)        │
 * │                                │                   │
 * ├────────────────────────────────┴───────────────────┤
 * │ PlaybackControls (future)                          │
 * └────────────────────────────────────────────────────┘
 */
export default function App() {
  const WEBSOCKET_URL = 'ws://localhost:8765/stream'

  // WebSocket connection
  const { latestFrame, connectionState, isConnected, reconnect } = useWebSocket(WEBSOCKET_URL, {
    autoReconnect: true,
    reconnectInterval: 3000,
  })

  // Playback state (Phase 3 MVP: non-functional, just for UI)
  const [isPaused, setIsPaused] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)

  return (
    <div style={appContainerStyle}>
      {/* Header */}
      <header style={headerStyle}>
        <h1 style={titleStyle}>🧭 nuPlan Real-Time Visualization</h1>
        <div style={headerRightStyle}>
          <ConnectionIndicator state={connectionState} />
          <button onClick={reconnect} style={reconnectButtonStyle}>
            Reconnect
          </button>
        </div>
      </header>

      {/* Main content area */}
      <div style={mainContentStyle}>
        {/* Left: Canvas */}
        <div style={canvasContainerStyle}>
          <SimulationCanvas
            frame={latestFrame}
            width={800}
            height={600}
            metersPerPixel={0.2}
          />
        </div>

        {/* Right: Debug panel */}
        <DebugPanel frame={latestFrame} connectionState={connectionState} />
      </div>

      {/* Bottom: Playback controls (Phase 3 MVP - placeholder) */}
      <PlaybackControls
        isPaused={isPaused}
        onPauseToggle={() => setIsPaused(!isPaused)}
        playbackSpeed={playbackSpeed}
        onSpeedChange={setPlaybackSpeed}
      />
    </div>
  )
}

/**
 * ConnectionIndicator - Visual indicator for WebSocket state.
 */
function ConnectionIndicator({ state }) {
  const stateConfig = {
    disconnected: { color: '#888', text: '● Disconnected' },
    connecting: { color: '#ffaa00', text: '● Connecting...' },
    connected: { color: '#00ff00', text: '● Connected' },
    error: { color: '#ff0000', text: '● Error' },
  }

  const config = stateConfig[state] || stateConfig.disconnected

  return (
    <div style={{ ...indicatorStyle, color: config.color }}>
      {config.text}
    </div>
  )
}

// Styles
const appContainerStyle = {
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  width: '100vw',
  backgroundColor: '#1e1e1e',
  color: '#ffffff',
  overflow: 'hidden',
}

const headerStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '12px 24px',
  backgroundColor: '#2a2a2a',
  borderBottom: '2px solid #444',
}

const titleStyle = {
  margin: 0,
  fontSize: '24px',
  fontFamily: 'monospace',
}

const headerRightStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '16px',
}

const indicatorStyle = {
  fontSize: '14px',
  fontFamily: 'monospace',
  fontWeight: 'bold',
}

const reconnectButtonStyle = {
  padding: '6px 12px',
  fontSize: '12px',
  fontWeight: 'bold',
  cursor: 'pointer',
  backgroundColor: '#444',
  color: '#fff',
  border: '1px solid #666',
  borderRadius: '4px',
  fontFamily: 'monospace',
}

const mainContentStyle = {
  display: 'flex',
  flex: 1,
  overflow: 'hidden',
}

const canvasContainerStyle = {
  flex: 1,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  padding: '24px',
  backgroundColor: '#1e1e1e',
}

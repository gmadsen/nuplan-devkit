// ABOUTME: Playback controls for simulation visualization (pause, speed, step)
// ABOUTME: Provides user controls for real-time playback manipulation (Phase 3 MVP - minimal implementation)

import React from 'react'

/**
 * PlaybackControls - Control panel for playback (future enhancement).
 *
 * Phase 3 MVP: Minimal implementation (just a placeholder).
 * Future: Add pause/resume, speed control, frame stepping, etc.
 *
 * Props:
 * - isPaused: Boolean indicating if playback is paused
 * - onPauseToggle: Callback for pause/resume
 * - playbackSpeed: Current playback speed multiplier (1.0 = normal)
 * - onSpeedChange: Callback for speed change
 */
export function PlaybackControls({
  isPaused = false,
  onPauseToggle = () => {},
  playbackSpeed = 1.0,
  onSpeedChange = () => {},
}) {
  return (
    <div style={controlsStyle}>
      <h4 style={titleStyle}>Playback Controls</h4>

      {/* Pause/Resume Button */}
      <div style={controlGroupStyle}>
        <button onClick={onPauseToggle} style={buttonStyle}>
          {isPaused ? '▶ Resume' : '⏸ Pause'}
        </button>
      </div>

      {/* Speed Control */}
      <div style={controlGroupStyle}>
        <label style={labelStyle}>Speed:</label>
        <button onClick={() => onSpeedChange(0.5)} style={smallButtonStyle}>0.5x</button>
        <button onClick={() => onSpeedChange(1.0)} style={smallButtonStyle}>1.0x</button>
        <button onClick={() => onSpeedChange(2.0)} style={smallButtonStyle}>2.0x</button>
        <span style={speedDisplayStyle}>{playbackSpeed.toFixed(1)}x</span>
      </div>

      {/* Note about Phase 3 limitations */}
      <div style={noteStyle}>
        Note: Playback controls are non-functional in Phase 3 MVP.
        Live streaming mode only (no buffering/stepping yet).
      </div>
    </div>
  )
}

// Styles
const controlsStyle = {
  padding: '12px',
  backgroundColor: '#2a2a2a',
  borderTop: '2px solid #444',
  borderBottom: '2px solid #444',
  fontFamily: 'monospace',
}

const titleStyle = {
  margin: '0 0 12px 0',
  fontSize: '14px',
  color: '#00ffff',
}

const controlGroupStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '8px',
}

const buttonStyle = {
  padding: '8px 16px',
  fontSize: '14px',
  fontWeight: 'bold',
  cursor: 'pointer',
  backgroundColor: '#444',
  color: '#fff',
  border: '1px solid #666',
  borderRadius: '4px',
  fontFamily: 'monospace',
}

const smallButtonStyle = {
  ...buttonStyle,
  padding: '4px 8px',
  fontSize: '12px',
}

const labelStyle = {
  color: '#aaa',
  fontSize: '12px',
}

const speedDisplayStyle = {
  color: '#00ff00',
  fontSize: '12px',
  fontWeight: 'bold',
  marginLeft: '8px',
}

const noteStyle = {
  marginTop: '12px',
  padding: '8px',
  backgroundColor: '#333',
  borderRadius: '4px',
  fontSize: '11px',
  color: '#888',
  fontStyle: 'italic',
}

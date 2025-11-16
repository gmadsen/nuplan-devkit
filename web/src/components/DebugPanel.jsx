// ABOUTME: Debug panel component for displaying simulation metrics and state
// ABOUTME: Shows ego state, scenario info, connection status, and real-time statistics

import React from 'react'

/**
 * DebugPanel - Displays simulation state and metrics.
 *
 * Props:
 * - frame: Latest simulation frame from WebSocket
 * - connectionState: WebSocket connection state (disconnected | connecting | connected | error)
 */
export function DebugPanel({ frame, connectionState }) {
  return (
    <div style={panelStyle}>
      <h3 style={headerStyle}>Debug Panel</h3>

      {/* Connection Status */}
      <Section title="Connection">
        <StatusBadge state={connectionState} />
      </Section>

      {/* Scenario Info */}
      {frame?.type === 'simulation_start' && (
        <Section title="Scenario">
          <MetricRow label="Name" value={frame.data.scenario_name || 'N/A'} />
          <MetricRow label="Type" value={frame.data.scenario_type || 'N/A'} />
          <MetricRow label="Map" value={frame.data.map_name || 'N/A'} />
          <MetricRow label="Duration" value={`${frame.data.duration?.toFixed(1) || 'N/A'}s`} />
        </Section>
      )}

      {/* Simulation Step Info */}
      {frame?.type === 'simulation_step' && (
        <>
          <Section title="Simulation">
            <MetricRow label="Iteration" value={frame.data.iteration || 'N/A'} />
            <MetricRow label="Time" value={`${frame.data.time_s?.toFixed(2) || 'N/A'}s`} />
          </Section>

          <Section title="Ego State">
            <MetricRow
              label="Position"
              value={`(${frame.data.ego_state?.x?.toFixed(1) || 'N/A'}, ${frame.data.ego_state?.y?.toFixed(1) || 'N/A'})`}
            />
            <MetricRow
              label="Heading"
              value={`${(frame.data.ego_state?.heading * 180 / Math.PI)?.toFixed(1) || 'N/A'}°`}
            />
            <MetricRow
              label="Velocity"
              value={`${frame.data.ego_state?.velocity?.toFixed(2) || 'N/A'} m/s`}
              highlight={frame.data.ego_state?.velocity > 15}
            />
            <MetricRow
              label="Acceleration"
              value={`${frame.data.ego_state?.acceleration?.toFixed(2) || 'N/A'} m/s²`}
              highlight={Math.abs(frame.data.ego_state?.acceleration || 0) > 3}
            />
            <MetricRow
              label="Steering"
              value={`${(frame.data.ego_state?.steering_angle * 180 / Math.PI)?.toFixed(1) || 'N/A'}°`}
            />
          </Section>

          <Section title="Environment">
            <MetricRow label="Trajectory Points" value={frame.data.trajectory?.length || 0} />
            <MetricRow label="Tracked Objects" value={frame.data.tracked_objects?.length || 0} />
          </Section>
        </>
      )}

      {/* Simulation End Info */}
      {frame?.type === 'simulation_end' && (
        <Section title="Simulation Complete">
          <MetricRow label="Total Steps" value={frame.data.total_steps || 'N/A'} />
          <MetricRow label="Scenario" value={frame.data.scenario_name || 'N/A'} />
          <div style={completeStyle}>✓ Simulation Complete</div>
        </Section>
      )}

      {/* No Data */}
      {!frame && (
        <div style={noDataStyle}>
          Waiting for simulation data...
        </div>
      )}
    </div>
  )
}

/**
 * Section component for grouping metrics.
 */
function Section({ title, children }) {
  return (
    <div style={sectionStyle}>
      <h4 style={sectionTitleStyle}>{title}</h4>
      {children}
    </div>
  )
}

/**
 * MetricRow component for displaying a label-value pair.
 */
function MetricRow({ label, value, highlight = false }) {
  return (
    <div style={metricRowStyle}>
      <span style={labelStyle}>{label}:</span>
      <span style={{
        ...valueStyle,
        color: highlight ? '#ff6464' : '#00ff00',
      }}>
        {value}
      </span>
    </div>
  )
}

/**
 * StatusBadge component for connection state.
 */
function StatusBadge({ state }) {
  const stateColors = {
    disconnected: '#888',
    connecting: '#ffaa00',
    connected: '#00ff00',
    error: '#ff0000',
  }

  const stateLabels = {
    disconnected: 'Disconnected',
    connecting: 'Connecting...',
    connected: 'Connected',
    error: 'Error',
  }

  return (
    <div style={{
      ...badgeStyle,
      backgroundColor: stateColors[state] || '#888',
    }}>
      {stateLabels[state] || state}
    </div>
  )
}

// Styles
const panelStyle = {
  width: '300px',
  height: '100%',
  backgroundColor: '#2a2a2a',
  padding: '16px',
  overflowY: 'auto',
  borderLeft: '2px solid #444',
  fontFamily: 'monospace',
  fontSize: '12px',
}

const headerStyle = {
  margin: '0 0 16px 0',
  fontSize: '18px',
  color: '#ffffff',
  borderBottom: '2px solid #444',
  paddingBottom: '8px',
}

const sectionStyle = {
  marginBottom: '16px',
  padding: '8px',
  backgroundColor: '#333',
  borderRadius: '4px',
}

const sectionTitleStyle = {
  margin: '0 0 8px 0',
  fontSize: '14px',
  color: '#00ffff',
  fontWeight: 'bold',
}

const metricRowStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  marginBottom: '4px',
  padding: '2px 0',
}

const labelStyle = {
  color: '#aaa',
}

const valueStyle = {
  color: '#00ff00',
  fontWeight: 'bold',
}

const badgeStyle = {
  display: 'inline-block',
  padding: '4px 12px',
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: 'bold',
  color: '#000',
}

const noDataStyle = {
  textAlign: 'center',
  color: '#888',
  marginTop: '40px',
  fontSize: '14px',
}

const completeStyle = {
  textAlign: 'center',
  color: '#00ff00',
  fontSize: '16px',
  fontWeight: 'bold',
  marginTop: '8px',
}

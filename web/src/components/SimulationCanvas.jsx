// ABOUTME: Main bird's-eye view canvas component for real-time simulation visualization
// ABOUTME: Renders ego vehicle, planned trajectory, and tracked agents using 2D canvas

import React, { useRef, useEffect, useState } from 'react'
import { CoordinateTransformer, computeVehicleCorners } from '../utils/coordinateTransform'
import {
  drawOrientedBox,
  drawVelocityVector,
  drawPolyline,
  drawCircle,
  drawText,
  getObjectTypeColor,
} from '../utils/geometryHelpers'

/**
 * SimulationCanvas - Bird's-eye view rendering of simulation state.
 *
 * Props:
 * - frame: Latest simulation frame from WebSocket (simulation_step message)
 * - width: Canvas width in pixels (default: 800)
 * - height: Canvas height in pixels (default: 600)
 * - metersPerPixel: Initial scale factor (default: 0.2)
 */
export function SimulationCanvas({ frame, width = 800, height = 600, metersPerPixel = 0.2 }) {
  const canvasRef = useRef(null)
  const transformerRef = useRef(null)
  const [zoom, setZoom] = useState(metersPerPixel)

  // Initialize coordinate transformer
  useEffect(() => {
    if (!transformerRef.current) {
      transformerRef.current = new CoordinateTransformer(width, height, zoom)
    } else {
      transformerRef.current.updateDimensions(width, height)
      transformerRef.current.setScale(zoom)
    }
  }, [width, height, zoom])

  // Render frame
  useEffect(() => {
    if (!frame || !canvasRef.current || !transformerRef.current) {
      return
    }

    // Only render simulation_step messages
    if (frame.type !== 'simulation_step') {
      return
    }

    const ctx = canvasRef.current.getContext('2d')
    const transformer = transformerRef.current
    const data = frame.data

    // Clear canvas
    ctx.fillStyle = '#1e1e1e'
    ctx.fillRect(0, 0, width, height)

    // Update transformer center to follow ego
    if (data.ego_state) {
      transformer.setCenter(data.ego_state.x, data.ego_state.y)
    }

    // Render grid (optional, for reference)
    renderGrid(ctx, transformer, width, height)

    // Render trajectory
    if (data.trajectory && data.trajectory.length > 0) {
      renderTrajectory(ctx, transformer, data.trajectory)
    }

    // Render tracked objects
    if (data.tracked_objects && data.tracked_objects.length > 0) {
      renderTrackedObjects(ctx, transformer, data.tracked_objects)
    }

    // Render ego vehicle (on top)
    if (data.ego_state) {
      renderEgoVehicle(ctx, transformer, data.ego_state)
    }

    // Render iteration counter
    if (data.iteration !== undefined) {
      renderIterationInfo(ctx, data.iteration, data.time_s, width, height)
    }
  }, [frame, width, height, zoom])

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          border: '2px solid #444',
          borderRadius: '4px',
          backgroundColor: '#1e1e1e',
        }}
      />
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '8px 12px',
        borderRadius: '4px',
        fontSize: '12px',
      }}>
        <div>Zoom: {(1 / zoom).toFixed(1)}x</div>
        <button onClick={() => setZoom(z => z * 1.2)} style={buttonStyle}>-</button>
        <button onClick={() => setZoom(z => z / 1.2)} style={buttonStyle}>+</button>
        <button onClick={() => setZoom(0.2)} style={buttonStyle}>Reset</button>
      </div>
    </div>
  )
}

const buttonStyle = {
  margin: '0 4px',
  padding: '4px 8px',
  fontSize: '12px',
  cursor: 'pointer',
  backgroundColor: '#333',
  color: '#fff',
  border: '1px solid #555',
  borderRadius: '3px',
}

/**
 * Render reference grid.
 */
function renderGrid(ctx, transformer, width, height) {
  const gridSpacing = 10  // 10 meters
  const pixelSpacing = transformer.worldDistanceToCanvas(gridSpacing)

  ctx.save()
  ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)'
  ctx.lineWidth = 1

  // Vertical lines
  for (let x = 0; x < width; x += pixelSpacing) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }

  // Horizontal lines
  for (let y = 0; y < height; y += pixelSpacing) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }

  ctx.restore()
}

/**
 * Render ego vehicle.
 */
function renderEgoVehicle(ctx, transformer, egoState) {
  // Ego vehicle dimensions (approximate)
  const length = 4.5  // meters
  const width = 2.0   // meters

  // Compute corners in world coordinates
  const corners = computeVehicleCorners(
    egoState.x,
    egoState.y,
    egoState.heading,
    length,
    width
  )

  // Convert to canvas coordinates
  const canvasCorners = corners.map(corner => transformer.worldToCanvas(corner.x, corner.y))

  // Draw ego vehicle (green)
  drawOrientedBox(ctx, canvasCorners, {
    fillColor: 'rgba(0, 255, 0, 0.4)',
    strokeColor: '#00ff00',
    lineWidth: 3,
  })

  // Draw velocity vector
  const egoCanvasPos = transformer.worldToCanvas(egoState.x, egoState.y)
  const velocityLength = transformer.worldDistanceToCanvas(egoState.velocity * 0.5)  // Scale for visibility

  drawVelocityVector(
    ctx,
    egoCanvasPos.x,
    egoCanvasPos.y,
    egoState.heading,
    velocityLength,
    {
      color: '#ffff00',
      lineWidth: 2,
      arrowHeadSize: 8,
    }
  )

  // Draw label
  drawText(ctx, 'EGO', egoCanvasPos.x, egoCanvasPos.y - 30, {
    font: 'bold 14px monospace',
    fillColor: '#00ff00',
  })
}

/**
 * Render planned trajectory.
 */
function renderTrajectory(ctx, transformer, trajectory) {
  // Convert trajectory points to canvas coordinates
  const canvasPoints = trajectory.map(point =>
    transformer.worldToCanvas(point.x, point.y)
  )

  // Draw trajectory polyline
  drawPolyline(ctx, canvasPoints, {
    strokeColor: '#00ffff',
    lineWidth: 2,
    lineDash: [5, 5],  // Dashed line
  })

  // Draw waypoint markers (every 5th point)
  trajectory.forEach((point, i) => {
    if (i % 5 === 0) {
      const canvasPoint = transformer.worldToCanvas(point.x, point.y)
      drawCircle(ctx, canvasPoint.x, canvasPoint.y, 3, {
        fillColor: 'rgba(0, 255, 255, 0.6)',
        strokeColor: '#00ffff',
        lineWidth: 1,
      })
    }
  })
}

/**
 * Render tracked objects (agents).
 */
function renderTrackedObjects(ctx, transformer, objects) {
  objects.forEach(obj => {
    // Compute corners
    const corners = computeVehicleCorners(
      obj.x,
      obj.y,
      obj.heading,
      obj.length || 4.5,  // Default if missing
      obj.width || 2.0
    )

    // Convert to canvas coordinates
    const canvasCorners = corners.map(corner => transformer.worldToCanvas(corner.x, corner.y))

    // Get color based on object type
    const colors = getObjectTypeColor(obj.type)

    // Draw object
    drawOrientedBox(ctx, canvasCorners, {
      fillColor: colors.fillColor,
      strokeColor: colors.strokeColor,
      lineWidth: 2,
    })

    // Draw ID label (first 6 chars)
    const canvasPos = transformer.worldToCanvas(obj.x, obj.y)
    const shortId = obj.id.substring(0, 6)
    drawText(ctx, shortId, canvasPos.x, canvasPos.y, {
      font: '10px monospace',
      fillColor: colors.strokeColor,
    })
  })
}

/**
 * Render iteration info overlay.
 */
function renderIterationInfo(ctx, iteration, timeS, width, height) {
  drawText(
    ctx,
    `Step: ${iteration} | Time: ${timeS.toFixed(1)}s`,
    width / 2,
    20,
    {
      font: 'bold 16px monospace',
      fillColor: '#ffffff',
      textAlign: 'center',
      textBaseline: 'top',
    }
  )
}

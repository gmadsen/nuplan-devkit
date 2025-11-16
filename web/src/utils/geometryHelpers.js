// ABOUTME: Geometry rendering utilities for canvas drawing (boxes, polylines, vectors)
// ABOUTME: Provides reusable functions for visualizing vehicles, trajectories, and agents

/**
 * Draw an oriented bounding box (vehicle) on canvas.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array<{x: number, y: number}>} corners - 4 corner points (CCW from front-left)
 * @param {Object} style - Rendering style
 * @param {string} style.fillColor - Fill color (default: 'rgba(0, 255, 0, 0.3)')
 * @param {string} style.strokeColor - Stroke color (default: '#00ff00')
 * @param {number} style.lineWidth - Line width (default: 2)
 */
export function drawOrientedBox(ctx, corners, style = {}) {
  const {
    fillColor = 'rgba(0, 255, 0, 0.3)',
    strokeColor = '#00ff00',
    lineWidth = 2,
  } = style

  if (corners.length !== 4) {
    console.error('drawOrientedBox requires exactly 4 corners')
    return
  }

  ctx.save()

  // Draw filled box
  ctx.fillStyle = fillColor
  ctx.beginPath()
  ctx.moveTo(corners[0].x, corners[0].y)
  for (let i = 1; i < corners.length; i++) {
    ctx.lineTo(corners[i].x, corners[i].y)
  }
  ctx.closePath()
  ctx.fill()

  // Draw outline
  ctx.strokeStyle = strokeColor
  ctx.lineWidth = lineWidth
  ctx.stroke()

  ctx.restore()
}


/**
 * Draw a velocity vector arrow.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} startX - Arrow start X (canvas coords)
 * @param {number} startY - Arrow start Y (canvas coords)
 * @param {number} heading - Direction in radians
 * @param {number} length - Arrow length in pixels
 * @param {Object} style - Rendering style
 * @param {string} style.color - Arrow color (default: '#ffff00')
 * @param {number} style.lineWidth - Line width (default: 3)
 * @param {number} style.arrowHeadSize - Arrow head size in pixels (default: 10)
 */
export function drawVelocityVector(ctx, startX, startY, heading, length, style = {}) {
  const {
    color = '#ffff00',
    lineWidth = 3,
    arrowHeadSize = 10,
  } = style

  // Compute end point
  const endX = startX + Math.cos(heading) * length
  const endY = startY - Math.sin(heading) * length  // Canvas Y is inverted

  ctx.save()

  // Draw line
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.beginPath()
  ctx.moveTo(startX, startY)
  ctx.lineTo(endX, endY)
  ctx.stroke()

  // Draw arrow head
  const headAngle = Math.PI / 6  // 30 degrees
  const head1X = endX - Math.cos(heading - headAngle) * arrowHeadSize
  const head1Y = endY + Math.sin(heading - headAngle) * arrowHeadSize
  const head2X = endX - Math.cos(heading + headAngle) * arrowHeadSize
  const head2Y = endY + Math.sin(heading + headAngle) * arrowHeadSize

  ctx.beginPath()
  ctx.moveTo(endX, endY)
  ctx.lineTo(head1X, head1Y)
  ctx.moveTo(endX, endY)
  ctx.lineTo(head2X, head2Y)
  ctx.stroke()

  ctx.restore()
}


/**
 * Draw a polyline (trajectory).
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array<{x: number, y: number}>} points - Polyline points (canvas coords)
 * @param {Object} style - Rendering style
 * @param {string} style.strokeColor - Line color (default: '#00ffff')
 * @param {number} style.lineWidth - Line width (default: 2)
 * @param {Array<number>} style.lineDash - Line dash pattern (default: [])
 */
export function drawPolyline(ctx, points, style = {}) {
  const {
    strokeColor = '#00ffff',
    lineWidth = 2,
    lineDash = [],
  } = style

  if (points.length < 2) {
    return  // Need at least 2 points
  }

  ctx.save()

  ctx.strokeStyle = strokeColor
  ctx.lineWidth = lineWidth
  ctx.setLineDash(lineDash)

  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y)
  }
  ctx.stroke()

  ctx.restore()
}


/**
 * Draw a circle (for small objects or markers).
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} centerX - Circle center X (canvas coords)
 * @param {number} centerY - Circle center Y (canvas coords)
 * @param {number} radius - Circle radius in pixels
 * @param {Object} style - Rendering style
 * @param {string} style.fillColor - Fill color (default: 'rgba(255, 0, 0, 0.5)')
 * @param {string} style.strokeColor - Stroke color (default: '#ff0000')
 * @param {number} style.lineWidth - Line width (default: 1)
 */
export function drawCircle(ctx, centerX, centerY, radius, style = {}) {
  const {
    fillColor = 'rgba(255, 0, 0, 0.5)',
    strokeColor = '#ff0000',
    lineWidth = 1,
  } = style

  ctx.save()

  ctx.fillStyle = fillColor
  ctx.strokeStyle = strokeColor
  ctx.lineWidth = lineWidth

  ctx.beginPath()
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
  ctx.fill()
  ctx.stroke()

  ctx.restore()
}


/**
 * Draw text label.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {string} text - Text to render
 * @param {number} x - Text position X (canvas coords)
 * @param {number} y - Text position Y (canvas coords)
 * @param {Object} style - Rendering style
 * @param {string} style.font - Font specification (default: '12px monospace')
 * @param {string} style.fillColor - Text color (default: '#ffffff')
 * @param {string} style.textAlign - Text alignment (default: 'center')
 * @param {string} style.textBaseline - Text baseline (default: 'middle')
 */
export function drawText(ctx, text, x, y, style = {}) {
  const {
    font = '12px monospace',
    fillColor = '#ffffff',
    textAlign = 'center',
    textBaseline = 'middle',
  } = style

  ctx.save()

  ctx.font = font
  ctx.fillStyle = fillColor
  ctx.textAlign = textAlign
  ctx.textBaseline = textBaseline

  ctx.fillText(text, x, y)

  ctx.restore()
}


/**
 * Get color for tracked object type.
 *
 * @param {string} objectType - Object type (e.g., 'vehicle', 'pedestrian', 'bicycle')
 * @returns {{fillColor: string, strokeColor: string}} - Colors for rendering
 */
export function getObjectTypeColor(objectType) {
  const colorMap = {
    'vehicle': { fillColor: 'rgba(255, 100, 100, 0.3)', strokeColor: '#ff6464' },
    'pedestrian': { fillColor: 'rgba(100, 255, 100, 0.3)', strokeColor: '#64ff64' },
    'bicycle': { fillColor: 'rgba(100, 100, 255, 0.3)', strokeColor: '#6464ff' },
    'generic_object': { fillColor: 'rgba(200, 200, 200, 0.3)', strokeColor: '#c8c8c8' },
  }

  // Match by substring (handles "tracked_objects.types.vehicle" etc.)
  for (const [key, colors] of Object.entries(colorMap)) {
    if (objectType.toLowerCase().includes(key)) {
      return colors
    }
  }

  // Default: gray
  return { fillColor: 'rgba(150, 150, 150, 0.3)', strokeColor: '#969696' }
}

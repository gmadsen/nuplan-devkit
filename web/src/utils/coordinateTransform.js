// ABOUTME: Coordinate transformation utilities for world-space to canvas-space conversion
// ABOUTME: Handles scaling, translation, and rotation for bird's-eye view rendering

/**
 * Coordinate transformer for converting world coordinates (meters) to canvas pixels.
 *
 * The transformer maintains:
 * - Center point (world coordinates to center in view)
 * - Scale (meters per pixel)
 * - Canvas dimensions (width, height)
 *
 * Coordinate systems:
 * - World: (x, y) in meters, origin at scenario start, heading in radians (0 = East, CCW)
 * - Canvas: (x, y) in pixels, origin at top-left, y-axis inverted (down = positive)
 */
export class CoordinateTransformer {
  /**
   * @param {number} canvasWidth - Canvas width in pixels
   * @param {number} canvasHeight - Canvas height in pixels
   * @param {number} metersPerPixel - Scale factor (default: 0.2 = 5 pixels per meter)
   */
  constructor(canvasWidth, canvasHeight, metersPerPixel = 0.2) {
    this.canvasWidth = canvasWidth
    this.canvasHeight = canvasHeight
    this.metersPerPixel = metersPerPixel

    // Center of view in world coordinates (updated dynamically)
    this.centerX = 0
    this.centerY = 0
  }

  /**
   * Update center point (typically to follow ego vehicle).
   *
   * @param {number} worldX - X coordinate in world frame (meters)
   * @param {number} worldY - Y coordinate in world frame (meters)
   */
  setCenter(worldX, worldY) {
    this.centerX = worldX
    this.centerY = worldY
  }

  /**
   * Update scale factor.
   *
   * @param {number} metersPerPixel - New scale (smaller = more zoomed in)
   */
  setScale(metersPerPixel) {
    this.metersPerPixel = metersPerPixel
  }

  /**
   * Convert world coordinates to canvas coordinates.
   *
   * @param {number} worldX - X in meters
   * @param {number} worldY - Y in meters
   * @returns {{x: number, y: number}} - Canvas coordinates (pixels)
   */
  worldToCanvas(worldX, worldY) {
    // Translate relative to center
    const relativeX = worldX - this.centerX
    const relativeY = worldY - this.centerY

    // Scale to pixels
    const pixelX = relativeX / this.metersPerPixel
    const pixelY = relativeY / this.metersPerPixel

    // Translate to canvas center and flip Y axis
    const canvasX = this.canvasWidth / 2 + pixelX
    const canvasY = this.canvasHeight / 2 - pixelY  // Flip Y (canvas Y grows downward)

    return { x: canvasX, y: canvasY }
  }

  /**
   * Convert canvas coordinates to world coordinates.
   *
   * @param {number} canvasX - X in pixels
   * @param {number} canvasY - Y in pixels
   * @returns {{x: number, y: number}} - World coordinates (meters)
   */
  canvasToWorld(canvasX, canvasY) {
    // Translate from canvas center
    const pixelX = canvasX - this.canvasWidth / 2
    const pixelY = -(canvasY - this.canvasHeight / 2)  // Flip Y

    // Scale to meters
    const relativeX = pixelX * this.metersPerPixel
    const relativeY = pixelY * this.metersPerPixel

    // Translate to world coordinates
    const worldX = this.centerX + relativeX
    const worldY = this.centerY + relativeY

    return { x: worldX, y: worldY }
  }

  /**
   * Convert world distance (meters) to canvas distance (pixels).
   *
   * @param {number} worldDistance - Distance in meters
   * @returns {number} - Distance in pixels
   */
  worldDistanceToCanvas(worldDistance) {
    return worldDistance / this.metersPerPixel
  }

  /**
   * Convert canvas distance (pixels) to world distance (meters).
   *
   * @param {number} canvasDistance - Distance in pixels
   * @returns {number} - Distance in meters
   */
  canvasDistanceToWorld(canvasDistance) {
    return canvasDistance * this.metersPerPixel
  }

  /**
   * Update canvas dimensions (e.g., on window resize).
   *
   * @param {number} width - New canvas width
   * @param {number} height - New canvas height
   */
  updateDimensions(width, height) {
    this.canvasWidth = width
    this.canvasHeight = height
  }
}


/**
 * Rotate a point around the origin.
 *
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 * @param {number} angle - Rotation angle in radians (CCW)
 * @returns {{x: number, y: number}} - Rotated coordinates
 */
export function rotatePoint(x, y, angle) {
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)

  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  }
}


/**
 * Compute oriented bounding box corners for a vehicle.
 *
 * @param {number} centerX - Vehicle center X (world coordinates)
 * @param {number} centerY - Vehicle center Y (world coordinates)
 * @param {number} heading - Vehicle heading in radians (0 = East, CCW)
 * @param {number} length - Vehicle length in meters
 * @param {number} width - Vehicle width in meters
 * @returns {Array<{x: number, y: number}>} - 4 corner points in world coordinates (CCW from front-left)
 */
export function computeVehicleCorners(centerX, centerY, heading, length, width) {
  // Local coordinates (relative to vehicle center, before rotation)
  const halfLength = length / 2
  const halfWidth = width / 2

  const corners = [
    { x: halfLength, y: halfWidth },   // Front-left
    { x: halfLength, y: -halfWidth },  // Front-right
    { x: -halfLength, y: -halfWidth }, // Rear-right
    { x: -halfLength, y: halfWidth },  // Rear-left
  ]

  // Rotate and translate to world coordinates
  return corners.map(({ x, y }) => {
    const rotated = rotatePoint(x, y, heading)
    return {
      x: centerX + rotated.x,
      y: centerY + rotated.y,
    }
  })
}

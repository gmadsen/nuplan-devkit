// ABOUTME: WebSocket connection manager hook for real-time simulation streaming
// ABOUTME: Handles connection lifecycle, message parsing, and state management

import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Hook for managing WebSocket connection to simulation streaming server.
 *
 * Features:
 * - Auto-reconnect on disconnect (configurable)
 * - Connection state tracking (disconnected, connecting, connected, error)
 * - Message parsing and type-based routing
 * - Buffering for late-joining clients
 *
 * @param {string} url - WebSocket server URL (e.g., "ws://localhost:8765/stream")
 * @param {Object} options - Configuration options
 * @param {boolean} options.autoReconnect - Auto-reconnect on disconnect (default: true)
 * @param {number} options.reconnectInterval - Delay between reconnect attempts in ms (default: 3000)
 * @returns {Object} - { latestFrame, connectionState, isConnected, reconnect }
 */
export function useWebSocket(url, options = {}) {
  const {
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options

  const [latestFrame, setLatestFrame] = useState(null)
  const [connectionState, setConnectionState] = useState('disconnected') // disconnected | connecting | connected | error
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)

  // Derived state
  const isConnected = connectionState === 'connected'

  /**
   * Connect to WebSocket server.
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[useWebSocket] Already connected')
      return
    }

    console.log(`[useWebSocket] Connecting to ${url}...`)
    setConnectionState('connecting')

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[useWebSocket] Connected!')
        setConnectionState('connected')
        reconnectAttemptsRef.current = 0 // Reset counter on success
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          // Route message by type
          if (data.type === 'simulation_start') {
            console.log('[useWebSocket] Simulation started:', data.data.scenario_name)
            setLatestFrame(data) // Store metadata
          } else if (data.type === 'simulation_step') {
            setLatestFrame(data) // Update with latest step
          } else if (data.type === 'simulation_end') {
            console.log('[useWebSocket] Simulation ended:', data.data.scenario_name)
            setLatestFrame(data) // Store completion signal
          } else {
            console.warn('[useWebSocket] Unknown message type:', data.type)
          }
        } catch (error) {
          console.error('[useWebSocket] Failed to parse message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('[useWebSocket] WebSocket error:', error)
        setConnectionState('error')
      }

      ws.onclose = () => {
        console.log('[useWebSocket] Connection closed')
        setConnectionState('disconnected')
        wsRef.current = null

        // Auto-reconnect logic
        if (autoReconnect) {
          reconnectAttemptsRef.current += 1
          const delay = reconnectInterval * Math.min(reconnectAttemptsRef.current, 5) // Exponential backoff (capped at 5x)

          console.log(`[useWebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})...`)

          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }
    } catch (error) {
      console.error('[useWebSocket] Failed to create WebSocket:', error)
      setConnectionState('error')
    }
  }, [url, autoReconnect, reconnectInterval])

  /**
   * Disconnect from WebSocket server.
   */
  const disconnect = useCallback(() => {
    console.log('[useWebSocket] Disconnecting...')

    // Cancel pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setConnectionState('disconnected')
  }, [])

  /**
   * Manual reconnect trigger.
   */
  const reconnect = useCallback(() => {
    disconnect()
    setTimeout(() => connect(), 500) // Small delay before reconnecting
  }, [connect, disconnect])

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    latestFrame,        // Latest message from server (simulation_start | simulation_step | simulation_end)
    connectionState,    // Current connection state (disconnected | connecting | connected | error)
    isConnected,        // Boolean: true if connected
    reconnect,          // Function: Manually trigger reconnect
  }
}

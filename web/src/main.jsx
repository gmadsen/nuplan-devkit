// ABOUTME: React application entry point for nuPlan real-time visualization
// ABOUTME: Renders root App component into DOM

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

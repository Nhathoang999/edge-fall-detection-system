import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const [status, setStatus] = useState("CONNECTING...");
  const [confidence, setConfidence] = useState(0.0);
  const [isFall, setIsFall] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    if (wsRef.current) wsRef.current.close();
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    wsRef.current = new WebSocket(`${wsProtocol}//${host}:8002/ws/video`);

    wsRef.current.onopen = () => {
      setStatus("CONNECTED");
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setImageSrc(`data:image/jpeg;base64,${data.image}`);
      
      const conf = (data.confidence * 100).toFixed(1);
      setConfidence(conf);
      
      if (data.label === "fall") {
        setIsFall(true);
        setStatus("🚨 FALL DETECTED! 🚨");
      } else {
        setIsFall(false);
        setStatus("SYSTEM SAFE");
      }
    };

    wsRef.current.onclose = () => {
      setStatus("DISCONNECTED");
      setIsFall(false);
    };
  };




  return (
    <div className={`dashboard-container ${isFall ? "alert-mode" : ""}`}>
      <header className="header">
        <h1>IoT Fall Detection Dashboard</h1>
        <div className={`status-badge ${isFall ? "danger" : "safe"}`}>
          {status} {status !== "CONNECTING..." && status !== "DISCONNECTED" && `(${confidence}%)`}
        </div>
      </header>
      
      <main className="main-content">
        <div className="video-container">
          {imageSrc ? (
            <img src={imageSrc} alt="Live Camera Feed" className="video-feed" />
          ) : (
            <div className="placeholder">
              <p>Waiting for stream...</p>
            </div>
          )}
          {isFall && <div className="overlay-alert">EMERGENCY</div>}
        </div>
        
        <aside className="sidebar">


          <div className="card">
            <h3>System Info</h3>
            <p><strong>Device:</strong> Edge Computing Node</p>
            <p><strong>Model:</strong> TFLite Skeleton Transformer</p>
            <p><strong>Source:</strong> Webcam (Local)</p>
          </div>

          <div className="card">
            <h3>Event Logs</h3>
            <p>Ready to monitor.</p>
            {isFall && <p className="text-danger">[{new Date().toLocaleTimeString()}] Fall Alert Triggered!</p>}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;

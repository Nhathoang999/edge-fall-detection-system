import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const [status, setStatus] = useState("CONNECTING...");
  const [confidence, setConfidence] = useState(0.0);
  const [isFall, setIsFall] = useState(false);
  const [videoSourceType, setVideoSourceType] = useState("webcam");
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

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("UPLOADING VIDEO...");
      const host = window.location.hostname;
      const response = await fetch(`http://${host}:8002/upload-video`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      console.log(result);
      setVideoSourceType("video");
      setStatus("PLAYING VIDEO");
    } catch (error) {
      console.error("Error uploading video", error);
      setStatus("UPLOAD FAILED");
    }
  };

  const handleResetCamera = async () => {
    try {
      const host = window.location.hostname;
      const response = await fetch(`http://${host}:8002/reset-camera`, {
        method: "POST"
      });
      const result = await response.json();
      console.log(result);
      setVideoSourceType("webcam");
      setStatus("SWITCHED TO WEBCAM");
    } catch (error) {
      console.error("Error resetting camera", error);
    }
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
          <div className="card controls-card">
            <h3>Video Source Controls</h3>
            <div className="controls-group">
              <label className={`btn-upload ${videoSourceType === "video" ? "active" : ""}`}>
                Upload Video Test (.mp4)
                <input type="file" accept="video/mp4,video/avi" onChange={handleFileUpload} hidden />
              </label>
              
              <button 
                className={`btn-webcam ${videoSourceType === "webcam" ? "active" : ""}`} 
                onClick={handleResetCamera}
              >
                Use Live Webcam
              </button>
            </div>
          </div>

          <div className="card">
            <h3>System Info</h3>
            <p><strong>Device:</strong> Edge Computing Node</p>
            <p><strong>Model:</strong> TFLite Skeleton Transformer</p>
            <p><strong>Source:</strong> {videoSourceType === "webcam" ? "Webcam (Local)" : "Uploaded Video"}</p>
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

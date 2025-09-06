import React, { useState, useRef, useEffect } from 'react';
import './App.css'; // We'll create this file for styling

function App() {
  const [predictions, setPredictions] = useState({});
  const [isStreaming, setIsStreaming] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webSocketRef = useRef(null);
  const intervalRef = useRef(null);

  const startStreaming = () => {
    setIsStreaming(true);
    // Connect to the WebSocket server
    webSocketRef.current = new WebSocket('ws://127.0.0.1:8000/predict/live');

    webSocketRef.current.onopen = () => {
      console.log("WebSocket connection established");
      
      // Start capturing frames from the webcam
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
        })
        .catch(err => {
          console.error("Error accessing webcam:", err);
          setIsStreaming(false); // Stop if webcam access is denied
        });
    };

    webSocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // We are now receiving the annotated image directly
      // And the probability dictionary
      if (data.annotated_image) {
        const imageElement = document.getElementById('processed-feed');
        imageElement.src = data.annotated_image;
      }
      if (data.probabilities) {
        setPredictions(data.probabilities);
      }
    };

    webSocketRef.current.onclose = () => {
      console.log("WebSocket connection closed");
      stopStreaming(); // Clean up if the connection closes
    };
    
    webSocketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      stopStreaming(); // Clean up on error
    };
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    if (webSocketRef.current) {
      webSocketRef.current.close();
    }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => {
    if (isStreaming) {
      // Send a frame to the backend every 200ms
      intervalRef.current = setInterval(() => {
        if (videoRef.current && canvasRef.current && webSocketRef.current?.readyState === WebSocket.OPEN) {
          const context = canvasRef.current.getContext('2d');
          context.drawImage(videoRef.current, 0, 0, 640, 480);
          const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8); // Get as base64
          webSocketRef.current.send(imageData);
        }
      }, 200);
    }

    // Cleanup function to stop everything when the component unmounts
    return () => {
      stopStreaming();
    };
  }, [isStreaming]);


  return (
    <div className="App">
      <header className="App-header">
        <h1>Live Facial Emotion Detector</h1>
        <p>Powered by FastAPI & React</p>
      </header>
      <div className="detector-container">
        <div className="video-container">
          <h3>Processed Feed</h3>
          {/* We will use a regular img tag to display the received annotated image */}
          <img id="processed-feed" alt="Live feed from the server" width="640" height="480" />
          
          {/* Hidden video and canvas elements for capturing */}
          <video ref={videoRef} width="640" height="480" autoPlay style={{ display: 'none' }}></video>
          <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
        </div>
        
        <div className="controls-container">
          <div className="buttons">
            <button onClick={startStreaming} disabled={isStreaming}>Start Webcam</button>
            <button onClick={stopStreaming} disabled={!isStreaming}>Stop Webcam</button>
          </div>
          <div className="predictions">
            <h3>Emotion Probabilities</h3>
            <ul>
              {Object.entries(predictions)
                .sort(([, a], [, b]) => b - a) // Sort by probability
                .map(([emotion, prob]) => (
                  <li key={emotion}>
                    <strong>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}:</strong>
                    <div className="progress-bar-container">
                      <div 
                        className="progress-bar" 
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
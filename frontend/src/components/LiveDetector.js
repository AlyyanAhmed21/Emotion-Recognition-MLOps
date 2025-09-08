import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, Paper, Grid, Alert, CircularProgress } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';

// A reusable component for showing the emotion bars
const PredictionsDisplay = ({ preds }) => (
    <Paper elevation={4} sx={{ p: 2, background: 'rgba(40,40,40,0.8)', height: '100%', minHeight: { xs: 'auto', md: '504px' } }}>
      <Typography variant="h6" gutterBottom sx={{color: '#03dac6'}}>Emotion Probabilities</Typography>
       {Object.keys(preds).length > 0 ? (
          <ul style={{ listStyleType: 'none', padding: 0 }}>
              {Object.entries(preds).sort(([, a], [, b]) => b - a).map(([emotion, prob]) => (
                  <li key={emotion} style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
                    <strong style={{ width: '100px', fontWeight: 400, textTransform: 'capitalize' }}>{emotion}:</strong>
                    <Box sx={{ flexGrow: 1, height: '20px', backgroundColor: '#333', borderRadius: '10px', mx: 2, overflow: 'hidden' }}>
                      <Box sx={{ height: '100%', backgroundColor: '#bb86fc', borderRadius: '10px', width: `${prob * 100}%`, transition: 'width 0.2s ease-in-out' }} />
                    </Box>
                    <span style={{ width: '50px', textAlign: 'right', fontWeight: 700, color: '#03dac6' }}>{(prob * 100).toFixed(1)}%</span>
                  </li>
              ))}
          </ul>
      ) : <Typography sx={{ color: '#888', fontStyle: 'italic', textAlign: 'center', mt: 4 }}>Waiting for prediction...</Typography>}
    </Paper>
);

const LiveDetector = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef(null);
  const isProcessing = useRef(false);

  const predictionLoop = async () => {
    // If stop button was clicked, this ref will be null, stopping the loop.
    if (!requestRef.current) return;

    if (videoRef.current && videoRef.current.readyState >= 3 && !isProcessing.current) {
      isProcessing.current = true;
      
      const context = canvasRef.current.getContext('2d');
      context.drawImage(videoRef.current, 0, 0, 640, 480);
      const imageData = canvasRef.current.toDataURL('image/jpeg', 0.7);

      try {
        const response = await fetch('http://127.0.0.1:8000/predict/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData }),
        });
        if (!response.ok) throw new Error('Network error');
        const data = await response.json();
        setProcessedImage(data.annotated_image);
        setPredictions(data.probabilities);
      } catch (err) {
        setError("Connection to server failed. Please ensure the backend is running.");
        setIsStreaming(false); // Stop streaming on error
      } finally {
        isProcessing.current = false;
      }
    }
    // Re-queue the next frame
    requestRef.current = requestAnimationFrame(predictionLoop);
  };
  
  const startStreaming = async () => {
    setError(null);
    setPredictions({});
    setIsStreaming(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.oncanplay = () => {
          // Start the prediction loop only when the video is ready to play
          requestRef.current = requestAnimationFrame(predictionLoop);
        };
      }
    } catch (err) {
      setError("Could not access webcam. Please check browser permissions.");
      setIsStreaming(false);
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false); // This will cause the loop to stop on its next check
    cancelAnimationFrame(requestRef.current);
    requestRef.current = null;

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setProcessedImage(null);
  };
  
  // A simple effect for cleanup if the user navigates away
  useEffect(() => {
    return () => stopStreaming();
  }, []); // Empty array means this runs only on component unmount

  return (
    <Box>
      <Box display="flex" justifyContent="center" gap={2} mb={3}>
        <Button variant="contained" onClick={startStreaming} disabled={isStreaming} startIcon={<VideocamIcon />}>Start Webcam</Button>
        <Button variant="outlined" color="secondary" onClick={stopStreaming} disabled={!isStreaming} startIcon={<VideocamOffIcon />}>Stop Webcam</Button>
      </Box>
      
      {error && <Alert severity="error" sx={{mb: 2}}>{error}</Alert>}
      
      <Grid container spacing={2} alignItems="stretch" justifyContent="center">
        <Grid item xs={12} lg={8}>
          <Box sx={{ border: '1px solid #333', borderRadius: 2, p: 1, backgroundColor: '#121212', width: '100%', aspectRatio: '4 / 3', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
            {/* The processed image from the backend */}
            {processedImage ? 
              <img src={processedImage} alt="Processed feed" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '4px', transform: 'scaleX(-1)' }}/> 
            :
            // The raw video feed (mirrored) - show this if streaming but no processed image yet
            (isStreaming ? 
                <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '4px', transform: 'scaleX(-1)' }}/>
                : <Typography sx={{ color: '#666' }}>Webcam feed will appear here</Typography>)
            }
          </Box>
        </Grid>
        <Grid item xs={12} lg={5}>
          <PredictionsDisplay preds={predictions} />
        </Grid>
      </Grid>
      
      <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
    </Box>
  );
};

export default LiveDetector;
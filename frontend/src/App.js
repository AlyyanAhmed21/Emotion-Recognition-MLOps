import React, { useState, useRef, useEffect } from 'react';
import { mockPredictLive } from './mockPredictor';
import ParticlesBackground from './ParticlesBackground';
import { Box, Container, Typography, Button, Tabs, Tab, Paper, Grid } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import MovieIcon from '@mui/icons-material/Movie';
import InfoIcon from '@mui/icons-material/Info';

function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [predictions, setPredictions] = useState({});
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const startStreaming = () => {
    setIsStreaming(true);
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(err => {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please check permissions.");
        setIsStreaming(false);
      });
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setAnnotatedImage(null);
    setPredictions({});
  };

  useEffect(() => {
    if (isStreaming) {
      intervalRef.current = setInterval(async () => {
        if (videoRef.current && canvasRef.current) {
          const context = canvasRef.current.getContext('2d');
          context.drawImage(videoRef.current, 0, 0, 640, 480);
          const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);
          const result = await mockPredictLive(imageData);
          if (result) {
            setAnnotatedImage(result.annotated_image);
            setPredictions(result.probabilities);
          }
        }
      }, 200);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isStreaming]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    // Stop the webcam stream if the user switches away from the live tab
    if (newValue !== 0 && isStreaming) {
        stopStreaming();
    }
  };

  const TabPanel = (props) => {
    const { children, value, index, ...other } = props;
    return (
      <div
        role="tabpanel"
        hidden={value !== index}
        id={`tabpanel-${index}`}
        aria-labelledby={`tab-${index}`}
        {...other}
      >
        {value === index && (
          <Box sx={{ p: 3 }}>
            {children}
          </Box>
        )}
      </div>
    );
  };

  return (
    <>
      <ParticlesBackground />
      <Container maxWidth="lg" sx={{ textAlign: 'center', position: 'relative', zIndex: 1, py: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 700, color: '#FFF' }}>
          Facial Emotion Detector
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ color: '#bb86fc', mb: 4 }}>
          {activeTab === 0 ? "Real-time analysis from your webcam" : "Upload an image or video to begin"}
        </Typography>

        <Paper elevation={8} sx={{ background: 'rgba(30, 30, 30, 0.8)', backdropFilter: 'blur(10px)', borderRadius: 4, p: 2 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} centered>
              <Tab icon={<VideocamIcon />} label="Live Detection" />
              <Tab icon={<PhotoCameraIcon />} label="Upload Image" />
              <Tab icon={<MovieIcon />} label="Upload Video" />
              <Tab icon={<InfoIcon />} label="About" />
            </Tabs>
          </Box>
          
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={4} alignItems="flex-start">
              <Grid item xs={12} md={7}>
                <Box sx={{ border: '1px solid #333', borderRadius: 2, p: 1, backgroundColor: '#121212' }}>
                  {annotatedImage ? 
                    <img src={annotatedImage} alt="Live feed" style={{ width: '100%', height: 'auto', borderRadius: '4px' }}/> :
                    <Box sx={{ width: '100%', paddingTop: '75%', position: 'relative', backgroundColor: '#000', borderRadius: '4px' }}>
                      <Typography sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#666' }}>Webcam feed will appear here</Typography>
                    </Box>
                  }
                </Box>
              </Grid>
              <Grid item xs={12} md={5}>
                 <Box display="flex" flexDirection="column" gap={2}>
                  <Box display="flex" justifyContent="center" gap={2}>
                    <Button variant="contained" onClick={startStreaming} disabled={isStreaming} startIcon={<VideocamIcon />}>Start Webcam</Button>
                    <Button variant="outlined" color="secondary" onClick={stopStreaming} disabled={!isStreaming} startIcon={<VideocamOffIcon />}>Stop Webcam</Button>
                  </Box>
                  <Paper elevation={4} sx={{ p: 2, background: 'rgba(40,40,40,0.8)' }}>
                    <Typography variant="h6" gutterBottom sx={{color: '#03dac6'}}>Emotion Probabilities</Typography>
                     {Object.keys(predictions).length > 0 ? (
                        <ul style={{ listStyleType: 'none', padding: 0 }}>
                            {Object.entries(predictions)
                            .sort(([, a], [, b]) => b - a)
                            .map(([emotion, prob]) => (
                                <li key={emotion} style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
                                <strong style={{ width: '100px', fontWeight: 400 }}>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}:</strong>
                                <Box sx={{ flexGrow: 1, height: '12px', backgroundColor: '#333', borderRadius: '6px', mx: 2, overflow: 'hidden' }}>
                                    <Box sx={{ height: '100%', backgroundColor: '#bb86fc', borderRadius: '6px', width: `${prob * 100}%`, transition: 'width 0.3s ease-in-out' }} />
                                </Box>
                                <span style={{ width: '50px', textAlign: 'right', fontWeight: 700, color: '#03dac6' }}>{(prob * 100).toFixed(1)}%</span>
                                </li>
                            ))}
                        </ul>
                    ) : <Typography sx={{ color: '#888', fontStyle: 'italic', textAlign: 'center', mt: 2 }}>Waiting for prediction...</Typography>}
                  </Paper>
                </Box>
              </Grid>
            </Grid>
          </TabPanel>
          <TabPanel value={activeTab} index={1}>
            <Typography>Image Upload Feature Coming Soon!</Typography>
          </TabPanel>
          <TabPanel value={activeTab} index={2}>
            <Typography>Video Upload Feature Coming Soon!</Typography>
          </TabPanel>
          <TabPanel value={activeTab} index={3}>
              <Box sx={{ textAlign: 'left', p: 2, lineHeight: 1.6 }}>
                  <Typography variant="h5" gutterBottom>About This Model</Typography>
                  <Typography variant="h6">Model Architecture</Typography>
                  <ul>
                      <li><Typography><strong>Base Model:</strong> MobileNetV2 (pre-trained on ImageNet)</Typography></li>
                      <li><Typography><strong>Classifier Head:</strong> A custom head with a Dense layer (128 neurons, ReLU, L2 regularization) and a Dropout layer (rate: 0.5) was added.</Typography></li>
                      <li><Typography><strong>Output:</strong> 7 emotion classes.</Typography></li>
                  </ul>
                  <Typography variant="h6">Dataset</Typography>
                  <Typography>The model was trained on a combined dataset of FER+ and CK+ for diversity and quality.</Typography>
                  <Typography variant="h6" sx={{mt: 2}}>Performance</Typography>
                  <Typography>The final model achieved a high validation accuracy, demonstrating strong generalization. Performance on "in-the-wild" faces may vary from the controlled data.</Typography>
              </Box>
          </TabPanel>
        </Paper>

        {/* Hidden elements for capturing video frames */}
        <video ref={videoRef} width="640" height="480" autoPlay muted style={{ display: 'none' }}></video>
        <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
      </Container>
    </>
  );
}

export default App;
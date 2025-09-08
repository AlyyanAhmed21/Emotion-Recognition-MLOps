import React from 'react';
import ParticlesBackground from './ParticlesBackground';
import { Box, Container, Typography, Tabs, Tab, Paper } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import MovieIcon from '@mui/icons-material/Movie';
import InfoIcon from '@mui/icons-material/Info';

// Import our new, clean components
import LiveDetector from './components/LiveDetector';
import ImageUploader from './components/ImageUploader';
import VideoUploader from './components/VideoUploader';
import AboutSection from './components/AboutSection';

// Helper to manage tab content
const TabPanel = (props) => {
    const { children, value, index } = props;
    return <div hidden={value !== index}>{value === index && <Box sx={{ p: 3 }}>{children}</Box>}</div>;
};

function App() {
  const [activeTab, setActiveTab] = React.useState(0);
  const handleTabChange = (event, newValue) => setActiveTab(newValue);
  
  return (
    <>
      <ParticlesBackground />
      <Container maxWidth="xl" sx={{ textAlign: 'center', position: 'relative', zIndex: 1, py: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom>Facial Emotion Detector</Typography>
        <Typography variant="h6" color="text.secondary" sx={{ color: '#bb86fc', mb: 4 }}>
          A real-time, AI-powered application using Vision Transformers
        </Typography>

        <Paper elevation={8} sx={{ background: 'rgba(30, 30, 30, 0.8)', backdropFilter: 'blur(10px)', borderRadius: 4, p: 2, minHeight: '600px' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} centered>
              <Tab icon={<VideocamIcon />} label="Live Detection" />
              <Tab icon={<PhotoCameraIcon />} label="Upload Image" />
              <Tab icon={<MovieIcon />} label="Upload Video" />
              <Tab icon={<InfoIcon />} label="About" />
            </Tabs>
          </Box>
          
          <TabPanel value={activeTab} index={0}><LiveDetector /></TabPanel>
          <TabPanel value={activeTab} index={1}><ImageUploader /></TabPanel>
          <TabPanel value={activeTab} index={2}><VideoUploader /></TabPanel>
          <TabPanel value={activeTab} index={3}><AboutSection /></TabPanel>

        </Paper>
      </Container>
    </>
  );
}

export default App;
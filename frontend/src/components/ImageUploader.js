import React, { useState } from 'react';
import { Box, Typography, Button, Paper, Grid, CircularProgress, Alert } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios'; // We will use axios for consistency

// Helper component for displaying predictions
const PredictionsDisplay = ({ preds }) => {
    return (
        <Paper elevation={4} sx={{ p: 2, background: 'rgba(40,40,40,0.8)', height: '100%' }}>
          <Typography variant="h6" gutterBottom sx={{color: '#03dac6'}}>Emotion Probabilities</Typography>
           {Object.keys(preds).length > 0 ? (
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                  {Object.entries(preds)
                  .sort(([, a], [, b]) => b - a)
                  .map(([emotion, prob]) => (
                      <li key={emotion} style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
                      <strong style={{ width: '100px', fontWeight: 400, textTransform: 'capitalize' }}>{emotion}:</strong>
                      <Box sx={{ flexGrow: 1, height: '12px', backgroundColor: '#333', borderRadius: '6px', mx: 2, overflow: 'hidden' }}>
                          <Box sx={{ height: '100%', backgroundColor: '#bb86fc', borderRadius: '6px', width: `${prob * 100}%`, transition: 'width 0.3s ease-in-out' }} />
                      </Box>
                      <span style={{ width: '50px', textAlign: 'right', fontWeight: 700, color: '#03dac6' }}>{(prob * 100).toFixed(1)}%</span>
                      </li>
                  ))}
              </ul>
          ) : <Typography sx={{ color: '#888', fontStyle: 'italic', textAlign: 'center', mt: 4 }}>Waiting for prediction...</Typography>}
        </Paper>
    );
};


const ImageUploader = () => {
    const [processedImage, setProcessedImage] = useState(null);
    const [predictions, setPredictions] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleImageUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);
        setProcessedImage(null);
        setPredictions({});

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://12.0.0.1:8000/predict/image', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            
            if (response.data) {
                setProcessedImage(response.data.annotated_image);
                if(response.data.probabilities) setPredictions(response.data.probabilities);
            }

        } catch (err) {
            setError("Failed to get prediction from the server. Is the backend running?");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Grid container spacing={2} justifyContent="center" alignItems="flex-start">
            <Grid item xs={12} md={6}>
                <Button component="label" fullWidth variant="contained" startIcon={<CloudUploadIcon />} disabled={isLoading}>
                    Upload Image
                    <input type="file" hidden accept="image/*" onChange={handleImageUpload} />
                </Button>
                {isLoading && <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}><CircularProgress /></Box>}
                {error && <Alert severity="error" sx={{mt: 2}}>{error}</Alert>}
                <Box sx={{ mt: 4, border: '1px solid #333', borderRadius: 2, p: 1, backgroundColor: '#121212', minHeight: '300px' }}>
                  {processedImage ? 
                    <img src={processedImage} alt="Processed" style={{ width: '100%', height: 'auto', borderRadius: '4px' }}/> :
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '300px', color: '#666' }}>
                        Result will appear here
                    </Box>
                  }
                </Box>
            </Grid>
            <Grid item xs={12} md={6}>
                <PredictionsDisplay preds={predictions} />
            </Grid>
        </Grid>
    );
};

export default ImageUploader;
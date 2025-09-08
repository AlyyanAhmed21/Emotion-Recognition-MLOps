import React, { useState } from 'react';
// --- THIS IS THE FIX ---
// Add 'Alert' to the list of imports.
// Remove 'CircularProgress' since we are using 'LinearProgress' for videos.
import { Box, Typography, Button, Paper, Grid, LinearProgress, Alert } from '@mui/material';
// --- END FIX ---
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';

const VideoUploader = () => {
    const [processedVideo, setProcessedVideo] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState(null);

    const handleVideoUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);
        setProcessedVideo(null);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:8000/predict/video', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                responseType: 'blob',
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(percentCompleted);
                }
            });

            const url = URL.createObjectURL(response.data);
            setProcessedVideo(url);

        } catch (err) {
            setError("Failed to process video. Is the backend running?");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Grid container spacing={2} justifyContent="center" alignItems="flex-start">
            <Grid item xs={12} md={6}>
                <Button component="label" fullWidth variant="contained" startIcon={<CloudUploadIcon />} disabled={isLoading}>
                    Upload Video
                    <input type="file" hidden accept="video/*" onChange={handleVideoUpload} />
                </Button>
                {isLoading && (
                    <Box sx={{ mt: 2, width: '100%' }}>
                        <Typography sx={{mb: 1}}>{uploadProgress < 100 ? `Uploading: ${uploadProgress}%` : "Processing on server..."}</Typography>
                        {uploadProgress < 100 ? 
                         <LinearProgress variant="determinate" value={uploadProgress} /> :
                         <LinearProgress /> // This is an indeterminate progress bar for the server processing
                        }
                    </Box>
                )}
                 {error && <Alert severity="error" sx={{mt: 2}}>{error}</Alert>}
            </Grid>
            <Grid item xs={12} md={6}>
                <Box sx={{ border: '1px solid #333', borderRadius: 2, p: 1, backgroundColor: '#121212' }}>
                  {processedVideo ? 
                    <video src={processedVideo} controls autoPlay muted style={{ width: '100%', height: 'auto', borderRadius: '4px' }}/> :
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '300px', color: '#666' }}>
                        Processed video will appear here
                    </Box>
                  }
                </Box>
            </Grid>
        </Grid>
    );
};

export default VideoUploader;
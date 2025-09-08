import React from 'react';
import { Box, Typography } from '@mui/material';

const AboutSection = () => (
    <Box sx={{ textAlign: 'left', p: 2, lineHeight: 1.6, maxWidth: '800px', margin: 'auto' }}>
      <Typography variant="h5" gutterBottom>About This Model</Typography>
      <Typography variant="body1">This application uses a state-of-the-art Vision Transformer model to perform real-time facial emotion recognition.</Typography>
      <Typography variant="h6" sx={{mt: 2}}>Model Architecture</Typography>
      <ul>
          <li><Typography><strong>Base Model:</strong> Swin Transformer (tiny)</Typography></li>
          <li><Typography><strong>Model Name:</strong> `PangPang/affectnet-swin-tiny-patch4-window7-224`</Typography></li>
          <li><Typography><strong>Output:</strong> 8 emotion classes, including Neutral and Contempt.</Typography></li>
      </ul>
      <Typography variant="h6" sx={{mt: 2}}>Dataset</Typography>
      <Typography>The model was pre-trained on **AffectNet**, the largest database of facial expressions "in the wild," containing over 400,000 manually annotated images. This allows it to generalize well to real-world, spontaneous expressions.</Typography>
    </Box>
);

export default AboutSection;
// This file simulates the Python backend for frontend development.

const MOCK_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'];

/**
 * Simulates the predict_live function from our Python backend.
 * @param {string} frame_b64 - A base64 encoded image frame.
 * @returns {Promise<object>} A promise that resolves to an object with an annotated_image and probabilities.
 */
export const mockPredictLive = (frame_b64) => {
    // We wrap this in a Promise to simulate a network delay
    return new Promise(resolve => {
        setTimeout(() => {
            // --- Simulate Predictions ---
            // Create random-ish probabilities that look realistic
            let randomProbs = Array.from({ length: MOCK_CLASSES.length }, () => Math.random());
            const sum = randomProbs.reduce((a, b) => a + b, 0);
            randomProbs = randomProbs.map(p => p / sum); // Normalize to sum to 1

            const probabilities = MOCK_CLASSES.reduce((obj, key, index) => {
                obj[key] = randomProbs[index];
                return obj;
            }, {});
            
            // --- Simulate Annotated Image ---
            // For the mock, we just return the original frame without any annotations.
            // When we connect to the real backend, this will be the frame with the bounding box.
            const result = {
                annotated_image: frame_b64,
                probabilities: probabilities
            };

            resolve(result);
        }, 150); // Simulate a 150ms processing delay
    });
};
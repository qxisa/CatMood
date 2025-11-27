/**
 * CatMood - Cat Emotion Analyzer
 * Mobile-friendly web application for analyzing cat moods using camera
 */

/**
 * Cat mood data - combines mood classes with their associated emojis.
 * IMPORTANT: The order of moods must match the PyTorch model's class indices
 * for proper integration when using a real trained model.
 * Order matches: train_dataset.classes from the CatMoodelGC.ipynb notebook
 */
const MOOD_DATA = {
    'angry': { emoji: '😾', index: 0 },
    'curious': { emoji: '🙀', index: 1 },
    'hungry': { emoji: '😿', index: 2 },
    'playful': { emoji: '😸', index: 3 },
    'relaxed': { emoji: '😺', index: 4 },
    'sad': { emoji: '😢', index: 5 },
    'sick': { emoji: '🤒', index: 6 }
};

// Derived arrays for compatibility
const MOOD_CLASSES = Object.keys(MOOD_DATA);
const MOOD_EMOJIS = Object.fromEntries(
    Object.entries(MOOD_DATA).map(([mood, data]) => [mood, data.emoji])
);

// Simulation constants for demo mode
const SIMULATION_CONFIG = {
    CONFIDENCE_BOOST_MULTIPLIER: 1.5,  // Multiplier to boost top prediction confidence
    CONFIDENCE_BOOST_OFFSET: 0.2,       // Base offset added to top prediction
    MAX_CONFIDENCE: 0.95                // Maximum allowed confidence value
};

// DOM Elements
const elements = {
    camera: document.getElementById('camera'),
    canvas: document.getElementById('canvas'),
    preview: document.getElementById('preview'),
    placeholder: document.getElementById('placeholder'),
    startCameraBtn: document.getElementById('startCamera'),
    captureBtn: document.getElementById('captureBtn'),
    switchCameraBtn: document.getElementById('switchCamera'),
    fileInput: document.getElementById('fileInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resetBtn: document.getElementById('resetBtn'),
    results: document.getElementById('results'),
    loading: document.getElementById('loading'),
    moodEmoji: document.getElementById('moodEmoji'),
    primaryMood: document.getElementById('primaryMood'),
    confidenceBar: document.getElementById('confidenceBar'),
    confidence: document.getElementById('confidence'),
    predictionsList: document.getElementById('predictionsList')
};

// State
let stream = null;
let facingMode = 'environment'; // Start with back camera
let capturedImageData = null;

/**
 * Initialize the application
 */
function init() {
    setupEventListeners();
    checkCameraSupport();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    elements.startCameraBtn.addEventListener('click', startCamera);
    elements.captureBtn.addEventListener('click', capturePhoto);
    elements.switchCameraBtn.addEventListener('click', switchCamera);
    elements.fileInput.addEventListener('change', handleFileUpload);
    elements.analyzeBtn.addEventListener('click', analyzeMood);
    elements.resetBtn.addEventListener('click', reset);
}

/**
 * Check if camera is supported
 */
function checkCameraSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        elements.startCameraBtn.textContent = 'Camera Not Supported';
        elements.startCameraBtn.disabled = true;
        console.warn('Camera API not supported');
    }
}

/**
 * Start the camera
 */
async function startCamera() {
    try {
        // Stop any existing stream
        stopCamera();

        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 960 }
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.camera.srcObject = stream;
        
        // Update UI
        elements.placeholder.style.display = 'none';
        elements.camera.style.display = 'block';
        elements.preview.style.display = 'none';
        elements.startCameraBtn.style.display = 'none';
        elements.captureBtn.style.display = 'flex';
        elements.switchCameraBtn.style.display = 'flex';
        elements.analyzeBtn.style.display = 'none';
        elements.resetBtn.style.display = 'none';
        elements.results.style.display = 'none';

        // Mirror camera for front-facing
        elements.camera.style.transform = facingMode === 'user' ? 'scaleX(-1)' : 'scaleX(1)';

    } catch (error) {
        console.error('Error accessing camera:', error);
        handleCameraError(error);
    }
}

/**
 * Handle camera errors
 */
function handleCameraError(error) {
    let message = 'Unable to access camera. ';
    
    if (error.name === 'NotAllowedError') {
        message += 'Please grant camera permission and try again.';
    } else if (error.name === 'NotFoundError') {
        message += 'No camera found on this device.';
    } else if (error.name === 'NotReadableError') {
        message += 'Camera is already in use by another application.';
    } else {
        message += 'Please try again or upload an image instead.';
    }
    
    alert(message);
}

/**
 * Stop the camera stream
 */
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

/**
 * Switch between front and back camera
 */
async function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    await startCamera();
}

/**
 * Capture photo from camera
 */
function capturePhoto() {
    const canvas = elements.canvas;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = elements.camera.videoWidth;
    canvas.height = elements.camera.videoHeight;
    
    // Draw the video frame to canvas
    if (facingMode === 'user') {
        // Flip horizontally for front camera
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
    }
    context.drawImage(elements.camera, 0, 0);
    
    // Convert to image
    capturedImageData = canvas.toDataURL('image/jpeg', 0.9);
    
    // Show preview
    showPreview(capturedImageData);
    
    // Stop camera to save resources
    stopCamera();
}

/**
 * Handle file upload
 */
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        capturedImageData = e.target.result;
        showPreview(capturedImageData);
        stopCamera();
    };
    reader.readAsDataURL(file);
    
    // Reset file input
    event.target.value = '';
}

/**
 * Show image preview
 */
function showPreview(imageData) {
    elements.preview.src = imageData;
    elements.preview.style.display = 'block';
    elements.camera.style.display = 'none';
    elements.placeholder.style.display = 'none';
    
    // Update buttons
    elements.startCameraBtn.style.display = 'none';
    elements.captureBtn.style.display = 'none';
    elements.switchCameraBtn.style.display = 'none';
    elements.analyzeBtn.style.display = 'flex';
    elements.resetBtn.style.display = 'flex';
    elements.results.style.display = 'none';
}

/**
 * Reset to initial state
 */
function reset() {
    capturedImageData = null;
    
    // Reset UI
    elements.preview.style.display = 'none';
    elements.camera.style.display = 'none';
    elements.placeholder.style.display = 'flex';
    
    elements.startCameraBtn.style.display = 'flex';
    elements.captureBtn.style.display = 'none';
    elements.switchCameraBtn.style.display = 'none';
    elements.analyzeBtn.style.display = 'none';
    elements.resetBtn.style.display = 'none';
    elements.results.style.display = 'none';
    elements.loading.style.display = 'none';
}

/**
 * Analyze the cat's mood
 */
async function analyzeMood() {
    if (!capturedImageData) {
        alert('Please capture or upload an image first.');
        return;
    }
    
    // Show loading
    elements.loading.style.display = 'block';
    elements.analyzeBtn.style.display = 'none';
    
    try {
        // Simulate processing time (in real implementation, this would be model inference)
        const predictions = await runInference(capturedImageData);
        
        // Display results
        displayResults(predictions);
    } catch (error) {
        console.error('Error analyzing image:', error);
        alert('Error analyzing image. Please try again.');
        elements.analyzeBtn.style.display = 'flex';
    } finally {
        elements.loading.style.display = 'none';
    }
}

/**
 * Run model inference on the captured image
 * 
 * @param {string} imageData - Base64-encoded image data URL (format: 'data:image/jpeg;base64,...')
 * @returns {Promise<Array<{mood: string, confidence: number, emoji: string}>>} 
 *          Array of predictions sorted by confidence (descending), each containing:
 *          - mood: The predicted mood category name
 *          - confidence: Probability score between 0 and 1
 *          - emoji: Visual emoji representation of the mood
 * 
 * NOTE: This is currently a simulation for demonstration purposes.
 * To integrate a real model, replace simulatePredictions() with:
 * - TensorFlow.js: Load a converted model and run tf.model.predict()
 * - ONNX.js: Load an exported ONNX model and run inference
 */
async function runInference(imageData) {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate simulated predictions based on image analysis
    // In production, this would use a real model like TensorFlow.js
    const predictions = simulatePredictions();
    
    return predictions;
}

/**
 * Simulate model predictions
 * This creates realistic-looking predictions for demonstration
 */
function simulatePredictions() {
    // Generate random scores
    let scores = MOOD_CLASSES.map(() => Math.random());
    
    // Normalize to sum to 1
    const sum = scores.reduce((a, b) => a + b, 0);
    scores = scores.map(s => s / sum);
    
    // Create predictions array
    const predictions = MOOD_CLASSES.map((mood, index) => ({
        mood: mood,
        confidence: scores[index],
        emoji: MOOD_EMOJIS[mood]
    }));
    
    // Sort by confidence (descending)
    predictions.sort((a, b) => b.confidence - a.confidence);
    
    // Boost the top prediction for more realistic results
    // Using constants to make the simulation parameters clear and adjustable
    const { CONFIDENCE_BOOST_MULTIPLIER, CONFIDENCE_BOOST_OFFSET, MAX_CONFIDENCE } = SIMULATION_CONFIG;
    predictions[0].confidence = Math.min(
        predictions[0].confidence * CONFIDENCE_BOOST_MULTIPLIER + CONFIDENCE_BOOST_OFFSET,
        MAX_CONFIDENCE
    );
    
    // Re-normalize
    const newSum = predictions.reduce((a, b) => a + b.confidence, 0);
    predictions.forEach(p => p.confidence = p.confidence / newSum);
    
    return predictions;
}

/**
 * Display analysis results
 */
function displayResults(predictions) {
    const topPrediction = predictions[0];
    
    // Update primary mood display
    elements.moodEmoji.textContent = topPrediction.emoji;
    elements.primaryMood.textContent = topPrediction.mood;
    
    // Animate confidence bar
    const confidencePercent = Math.round(topPrediction.confidence * 100);
    elements.confidence.textContent = `${confidencePercent}% confidence`;
    
    // Use requestAnimationFrame for smooth animation
    requestAnimationFrame(() => {
        const fill = elements.confidenceBar.querySelector('.confidence-fill');
        fill.style.width = `${confidencePercent}%`;
    });
    
    // Display top 3 predictions
    elements.predictionsList.innerHTML = '';
    predictions.slice(0, 3).forEach((pred, index) => {
        const li = document.createElement('li');
        li.className = 'prediction-item';
        li.innerHTML = `
            <span class="prediction-rank">${index + 1}</span>
            <span class="prediction-mood">
                <span class="prediction-emoji">${pred.emoji}</span>
                ${pred.mood}
            </span>
            <span class="prediction-confidence">${Math.round(pred.confidence * 100)}%</span>
        `;
        elements.predictionsList.appendChild(li);
    });
    
    // Show results
    elements.results.style.display = 'block';
    elements.analyzeBtn.style.display = 'none';
    
    // Scroll to results on mobile
    elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', init);

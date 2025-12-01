/**
 * CatMood - Cat Emotion Analyzer
 * Mobile-friendly web application for analyzing cat moods using ONNX Runtime Web
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

// ONNX Model configuration
const MODEL_CONFIG = {
    MODEL_PATH: 'catmood_model.onnx',
    INPUT_SIZE: 224,
    // ImageNet normalization values (same as used in PyTorch training)
    MEAN: [0.485, 0.456, 0.406],
    STD: [0.229, 0.224, 0.225]
};

// ONNX Runtime session (loaded once)
let onnxSession = null;
let modelLoadError = null;

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
async function init() {
    setupEventListeners();
    checkCameraSupport();
    await loadModel();
}

/**
 * Load the ONNX model
 */
async function loadModel() {
    // Check if ONNX Runtime is available
    if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime Web not loaded. Model inference will not be available.');
        modelLoadError = new Error('ONNX Runtime Web library not loaded. This could be due to network issues, ad blockers, or browser security settings.');
        return;
    }
    
    try {
        console.log('Loading ONNX model...');
        onnxSession = await ort.InferenceSession.create(MODEL_CONFIG.MODEL_PATH, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log('Model loaded successfully');
        console.log('Model inputs:', onnxSession.inputNames);
        console.log('Model outputs:', onnxSession.outputNames);
    } catch (error) {
        console.error('Failed to load ONNX model:', error);
        // Provide more context about the error
        if (error.message.includes('no such file') || error.message.includes('404') || error.message.includes('Failed to fetch')) {
            modelLoadError = new Error('Model file not found. Please ensure catmood_model.onnx is uploaded to the repository and deployed with the website.');
        } else {
            modelLoadError = error;
        }
        // Model will be loaded on-demand if initial load fails
    }
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
        // Run ONNX model inference
        const predictions = await runInference(capturedImageData);
        
        // Display results
        displayResults(predictions);
    } catch (error) {
        console.error('Error analyzing image:', error);
        let errorMessage = 'Unable to analyze image. ';
        
        if (error.message.includes('Model not available') || error.message.includes('Failed to load model')) {
            errorMessage += 'The AI model could not be loaded. Please ensure the catmood_model.onnx file is available on the server.';
        } else if (error.message.includes('ONNX Runtime')) {
            errorMessage += 'The ONNX Runtime library failed to load. Please check your internet connection and try reloading the page.';
        } else {
            errorMessage += 'Please try again or use a different image.';
        }
        
        alert(errorMessage);
        elements.analyzeBtn.style.display = 'flex';
    } finally {
        elements.loading.style.display = 'none';
    }
}

/**
 * Run model inference on the captured image using ONNX Runtime Web
 * 
 * @param {string} imageData - Base64-encoded image data URL (format: 'data:image/jpeg;base64,...')
 * @returns {Promise<Array<{mood: string, confidence: number, emoji: string}>>} 
 *          Array of predictions sorted by confidence (descending), each containing:
 *          - mood: The predicted mood category name
 *          - confidence: Probability score between 0 and 1
 *          - emoji: Visual emoji representation of the mood
 */
async function runInference(imageData) {
    // Check if ONNX Runtime is available
    if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime Web library not loaded. Please check your internet connection and reload the page.');
    }
    
    // Load model if not already loaded
    if (!onnxSession) {
        try {
            await loadModel();
        } catch (error) {
            throw new Error('Failed to load model: ' + error.message);
        }
    }
    
    if (!onnxSession) {
        const errorMsg = modelLoadError 
            ? `Failed to load model: ${modelLoadError.message}`
            : 'Model not available. Please ensure catmood_model.onnx is uploaded to the repository and deployed with the website.';
        throw new Error(errorMsg);
    }
    
    // Preprocess the image
    const inputTensor = await preprocessImage(imageData);
    
    // Run inference
    const feeds = { [onnxSession.inputNames[0]]: inputTensor };
    const results = await onnxSession.run(feeds);
    
    // Get output tensor
    const outputData = results[onnxSession.outputNames[0]].data;
    
    // Apply softmax to get probabilities
    const probabilities = softmax(Array.from(outputData));
    
    // Create predictions array
    const predictions = MOOD_CLASSES.map((mood, index) => ({
        mood: mood,
        confidence: probabilities[index],
        emoji: MOOD_EMOJIS[mood]
    }));
    
    // Sort by confidence (descending)
    predictions.sort((a, b) => b.confidence - a.confidence);
    
    return predictions;
}

/**
 * Preprocess image for model inference
 * Resizes to 224x224 and normalizes with ImageNet stats
 * 
 * @param {string} imageData - Base64-encoded image data URL
 * @returns {Promise<ort.Tensor>} - ONNX tensor ready for inference
 */
async function preprocessImage(imageData) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            try {
                // Create canvas for resizing
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = MODEL_CONFIG.INPUT_SIZE;
                canvas.height = MODEL_CONFIG.INPUT_SIZE;
                
                // Draw and resize image
                ctx.drawImage(img, 0, 0, MODEL_CONFIG.INPUT_SIZE, MODEL_CONFIG.INPUT_SIZE);
                
                // Get image data
                const imageDataObj = ctx.getImageData(0, 0, MODEL_CONFIG.INPUT_SIZE, MODEL_CONFIG.INPUT_SIZE);
                const pixels = imageDataObj.data;
                
                // Create Float32Array for model input (CHW format: channels, height, width)
                const inputSize = MODEL_CONFIG.INPUT_SIZE;
                const float32Data = new Float32Array(3 * inputSize * inputSize);
                
                // Convert from RGBA (HWC) to normalized RGB (CHW)
                for (let y = 0; y < inputSize; y++) {
                    for (let x = 0; x < inputSize; x++) {
                        const pixelIdx = (y * inputSize + x) * 4;
                        
                        // Normalize pixel values: (pixel/255 - mean) / std
                        const r = (pixels[pixelIdx] / 255.0 - MODEL_CONFIG.MEAN[0]) / MODEL_CONFIG.STD[0];
                        const g = (pixels[pixelIdx + 1] / 255.0 - MODEL_CONFIG.MEAN[1]) / MODEL_CONFIG.STD[1];
                        const b = (pixels[pixelIdx + 2] / 255.0 - MODEL_CONFIG.MEAN[2]) / MODEL_CONFIG.STD[2];
                        
                        // CHW format: channel * H * W + y * W + x
                        float32Data[0 * inputSize * inputSize + y * inputSize + x] = r;
                        float32Data[1 * inputSize * inputSize + y * inputSize + x] = g;
                        float32Data[2 * inputSize * inputSize + y * inputSize + x] = b;
                    }
                }
                
                // Create ONNX tensor with shape [1, 3, 224, 224]
                const tensor = new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);
                resolve(tensor);
            } catch (error) {
                reject(error);
            }
        };
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = imageData;
    });
}

/**
 * Apply softmax function to convert logits to probabilities
 * 
 * @param {number[]} logits - Raw model output values
 * @returns {number[]} - Probability distribution summing to 1
 */
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(x => x / sumExp);
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

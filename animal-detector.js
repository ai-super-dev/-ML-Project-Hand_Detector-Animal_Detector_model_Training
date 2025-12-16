/**
 * Animal Detector Web Component
 * Detects animals using custom trained models
 * Uses MediaPipe Tasks Vision API
 */
class AnimalDetector extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.mediaPipeReady = false;
    this.mobilenet = null; // MobileNet model for feature extraction
    
    // Training-related properties
    this.trainingData = [];
    this.trainedModel = null;
    this.isTrainingCameraOpen = false;
    
    // Model management
    this.savedModels = [];
    this.selectedModelId = null;
    this.selectedModelInfo = null;
    this.testVideo = null;
    this.testStream = null;
    this.isTestModeActive = false;
    this.testAnimationFrame = null;
    
    // Confidence threshold for predictions (default 0.3 = 30%)
    // If prediction confidence is below this, output "NONE"
    // Lower default to avoid false negatives with newly trained models
    this.confidenceThreshold = 0.3;
    this.showRawPredictions = false; // Bypass threshold to see raw predictions
    
    // Load training data from localStorage
    this.loadTrainingData();
  }

  connectedCallback() {
    this.render();
    this.initializeMediaPipe();
  }

  disconnectedCallback() {
    this.closeTrainingCamera();
    this.stopTestMode();
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          width: 100%;
          max-width: 640px;
          margin: 0 auto;
          box-sizing: border-box;
        }
        
        .container {
          position: relative;
          width: 100%;
          background: #000;
          border-radius: 8px;
          overflow: hidden;
        }
        
        video {
          width: 100%;
          height: auto;
          display: block;
          transform: scaleX(-1); /* Mirror for better UX */
        }
        
        canvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
        }
        
        .animal-display {
          position: absolute;
          top: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(0, 0, 0, 0.7);
          color: #fff;
          padding: 15px 30px;
          border-radius: 8px;
          font-size: 24px;
          font-weight: bold;
          text-transform: uppercase;
          font-family: Arial, sans-serif;
          min-width: 100px;
          text-align: center;
          transition: all 0.3s ease;
        }
        
        .status {
          padding: 10px;
          text-align: center;
          background: #f0f0f0;
          color: #333;
          font-family: Arial, sans-serif;
        }
        
        button {
          margin: 10px;
          padding: 10px 20px;
          font-size: 16px;
          cursor: pointer;
          border: none;
          border-radius: 4px;
          background: #4CAF50;
          color: white;
        }
        
        button:hover {
          background: #45a049;
        }
        
        button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }
        
        .training-section {
          margin-top: 20px;
          padding: 15px;
          background: #f9f9f9;
          border-radius: 8px;
          border: 2px solid #ddd;
        }
        
        .training-section h3 {
          margin-top: 0;
          color: #667eea;
          font-size: 18px;
        }
        
        .training-controls {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-bottom: 10px;
        }
        
        .training-controls-row {
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
        }
        
        .training-controls input[type="text"] {
          padding: 8px 12px;
          font-size: 14px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          flex: 1;
          min-width: 200px;
        }
        
        .training-controls button {
          margin: 0;
        }
        
        .training-controls input[type="file"] {
          display: none;
        }
        
        .train-btn {
          background: #9C27B0;
        }
        
        .train-btn:hover {
          background: #7B1FA2;
        }
        
        .upload-btn {
          background: #FF9800;
        }
        
        .upload-btn:hover {
          background: #F57C00;
        }
        
        .capture-btn {
          background: #FF9800;
          position: relative;
        }
        
        .capture-btn:hover {
          background: #F57C00;
        }
        
        .capture-btn:active {
          background: #E65100;
          transform: scale(0.95);
        }
        
        .clear-btn {
          background: #f44336;
        }
        
        .clear-btn:hover {
          background: #d32f2f;
        }
        
        .toggle-btn {
          background: #2196F3;
        }
        
        .toggle-btn:hover {
          background: #1976D2;
        }
        
        .toggle-btn.active {
          background: #4CAF50;
        }
        
        .training-stats {
          margin-top: 10px;
          padding: 10px;
          background: white;
          border-radius: 4px;
          font-size: 14px;
          color: #333;
        }
        
        .training-stats strong {
          color: #667eea;
        }
      </style>
      
      <div style="padding: 15px; background: white; border-radius: 8px; margin-bottom: 20px;">
        <div id="statusText" style="text-align: center; color: #667eea; font-weight: bold;">Loading MediaPipe...</div>
      </div>
      <div class="training-section">
        <h3>🎓 Training Mode</h3>
        <div class="training-controls">
          <div style="font-size: 12px; color: #d32f2f; margin-bottom: 10px; padding: 8px; background: #ffebee; border-radius: 4px; border-left: 3px solid #d32f2f;">
            <strong>💡 Tip:</strong> For better detection, train with background/empty images labeled as "background" or "none" to reduce false positives.
          </div>
          <div class="training-controls-row">
            <input type="text" id="animalTypeInput" placeholder="Enter animal name (e.g., cat, dog, bird, background)">
          </div>
          <div class="training-controls-row">
            <input type="file" id="fileInput" accept="image/*" multiple>
            <button id="uploadBtn" class="upload-btn">Upload from Local</button>
            <button id="openCameraBtn" class="capture-btn">Open Camera for Training</button>
            <button id="takeScreenshotBtn" class="capture-btn" style="display: none;">Take Screenshot</button>
            <button id="closeCameraBtn" class="clear-btn" style="display: none;">Close Camera</button>
            <button id="clearBtn" class="clear-btn">Clear Training Data</button>
          </div>
        </div>
        <div id="trainingCameraContainer" style="display: none; margin-top: 15px; text-align: center;">
          <video id="trainingVideo" autoplay playsinline style="width: 100%; max-width: 640px; border-radius: 8px; background: #000; transform: scaleX(-1);"></video>
        </div>
        <div class="training-stats" id="trainingStats">
          <strong>Training Data:</strong> 0 samples
        </div>
        <div id="trainingDataList" style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px; max-height: 400px; overflow-y: auto;">
          <h4 style="margin-top: 0; color: #667eea;">Training Data List:</h4>
          <div id="trainingDataTable" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; min-height: 50px;">
            <p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">No training data yet. Upload images or take screenshots to get started.</p>
          </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px;">
          <label for="modelNameInput" style="display: block; margin-bottom: 5px; font-weight: bold; color: #667eea;">Model Name:</label>
          <input type="text" id="modelNameInput" placeholder="Enter model name (e.g., MyAnimalModel)" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">
          <button id="trainBtn" class="train-btn" disabled style="width: 100%;">Train Model</button>
        </div>
      </div>
      
      <div class="training-section" style="margin-top: 20px;">
        <h3>📚 Saved Models</h3>
        <div id="modelsList" style="margin-bottom: 15px; max-height: 200px; overflow-y: auto; background: white; padding: 10px; border-radius: 4px;">
          <p style="color: #666; margin: 0; text-align: center;">No models saved yet. Train a model to get started.</p>
        </div>
      </div>
      
      <div class="training-section" style="margin-top: 20px;">
        <h3>🧪 Test Mode</h3>
        <div class="training-controls">
          <div class="training-controls-row" style="margin-bottom: 10px;">
            <label for="confidenceThreshold" style="font-weight: bold; color: #667eea; margin-right: 10px;">Confidence Threshold:</label>
            <input type="range" id="confidenceThreshold" min="0.1" max="0.9" step="0.05" value="0.3" style="flex: 1; min-width: 200px;">
            <span id="thresholdValue" style="min-width: 50px; text-align: center; font-weight: bold; color: #667eea;">0.30</span>
          </div>
          <div style="margin-bottom: 10px;">
            <label style="display: flex; align-items: center; gap: 8px; font-size: 14px; color: #333;">
              <input type="checkbox" id="showRawPredictions" style="width: 18px; height: 18px; cursor: pointer;">
              <span>Show raw predictions (bypass threshold - for debugging)</span>
            </label>
          </div>
          <div style="font-size: 12px; color: #666; margin-bottom: 10px; padding: 5px; background: #f0f0f0; border-radius: 4px;">
            Predictions below this threshold will show "NONE". Higher = more strict (fewer false positives).<br>
            <strong>Tip:</strong> Enable "Show raw predictions" to see actual confidence values even when below threshold.
          </div>
          <div class="training-controls-row">
            <button id="testBtn" class="toggle-btn" disabled>Start Test</button>
            <button id="stopTestBtn" class="clear-btn" style="display: none;">Stop Test</button>
          </div>
        </div>
        <div id="testCameraContainer" style="display: none; margin-top: 15px; text-align: center;">
          <div style="position: relative; display: inline-block; width: 100%; max-width: 640px;">
            <video id="testVideo" autoplay playsinline style="width: 100%; border-radius: 8px; background: #000; transform: scaleX(-1);"></video>
            <canvas id="testCanvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
          </div>
           <div id="testResultContainer" style="position: relative; margin-top: 10px; display: flex; gap: 10px; justify-content: center;">
             <div id="testAnimalDisplay" style="flex: 1; max-width: 200px; padding: 10px 15px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; background: rgba(0, 0, 0, 0.7); color: #fff; min-height: 45px; display: flex; align-items: center; justify-content: center;">
               <span id="testAnimalValue">--</span>
             </div>
             <div id="testConfidenceDisplay" style="flex: 1; max-width: 200px; padding: 10px 15px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; background: rgba(0, 0, 0, 0.7); color: #fff; min-height: 45px; display: flex; align-items: center; justify-content: center;">
               <span id="testConfidenceValue" style="font-family: 'Courier New', monospace;">--</span>
             </div>
           </div>
        </div>
        <div id="testStatus" style="margin-top: 10px; padding: 10px; background: white; border-radius: 4px; font-size: 14px; color: #333;">
          <strong>Status:</strong> Select a model from the list above to enable testing.
        </div>
      </div>
    `;

    // Get references to elements
    this.statusText = this.shadowRoot.getElementById('statusText');
    
    // Training UI elements
    this.uploadBtn = this.shadowRoot.getElementById('uploadBtn');
    this.openCameraBtn = this.shadowRoot.getElementById('openCameraBtn');
    this.takeScreenshotBtn = this.shadowRoot.getElementById('takeScreenshotBtn');
    this.closeCameraBtn = this.shadowRoot.getElementById('closeCameraBtn');
    this.trainBtn = this.shadowRoot.getElementById('trainBtn');
    this.clearBtn = this.shadowRoot.getElementById('clearBtn');
    this.animalTypeInput = this.shadowRoot.getElementById('animalTypeInput');
    this.fileInput = this.shadowRoot.getElementById('fileInput');
    this.trainingStats = this.shadowRoot.getElementById('trainingStats');
    this.trainingCameraContainer = this.shadowRoot.getElementById('trainingCameraContainer');
    this.trainingVideo = this.shadowRoot.getElementById('trainingVideo');
    this.modelNameInput = this.shadowRoot.getElementById('modelNameInput');
    this.trainingDataTable = this.shadowRoot.getElementById('trainingDataTable');
    
    // Model management UI elements
    this.modelsList = this.shadowRoot.getElementById('modelsList');
    
    // Test mode UI elements
    this.testBtn = this.shadowRoot.getElementById('testBtn');
    this.stopTestBtn = this.shadowRoot.getElementById('stopTestBtn');
    this.testCameraContainer = this.shadowRoot.getElementById('testCameraContainer');
    this.testVideo = this.shadowRoot.getElementById('testVideo');
    this.testCanvas = this.shadowRoot.getElementById('testCanvas');
    this.testCtx = this.testCanvas.getContext('2d');
    this.testResultContainer = this.shadowRoot.getElementById('testResultContainer');
    this.testAnimalDisplay = this.shadowRoot.getElementById('testAnimalDisplay');
    this.testAnimalValue = this.shadowRoot.getElementById('testAnimalValue');
    this.testConfidenceDisplay = this.shadowRoot.getElementById('testConfidenceDisplay');
    this.testConfidenceValue = this.shadowRoot.getElementById('testConfidenceValue');
    this.testStatus = this.shadowRoot.getElementById('testStatus');
    this.confidenceThresholdSlider = this.shadowRoot.getElementById('confidenceThreshold');
    this.thresholdValueDisplay = this.shadowRoot.getElementById('thresholdValue');
    this.showRawPredictionsCheckbox = this.shadowRoot.getElementById('showRawPredictions');
    
    // Setup confidence threshold slider
    if (this.confidenceThresholdSlider && this.thresholdValueDisplay) {
      this.confidenceThresholdSlider.value = this.confidenceThreshold;
      this.thresholdValueDisplay.textContent = this.confidenceThreshold.toFixed(2);
      this.confidenceThresholdSlider.addEventListener('input', (e) => {
        this.confidenceThreshold = parseFloat(e.target.value);
        this.thresholdValueDisplay.textContent = this.confidenceThreshold.toFixed(2);
      });
    }
    
    // Setup raw predictions checkbox
    if (this.showRawPredictionsCheckbox) {
      this.showRawPredictionsCheckbox.addEventListener('change', (e) => {
        this.showRawPredictions = e.target.checked;
      });
    }
    
    // Setup button handlers - placeholder functions for now
    this.uploadBtn.addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    this.openCameraBtn.addEventListener('click', () => this.openTrainingCamera());
    this.takeScreenshotBtn.addEventListener('click', () => this.takeScreenshot());
    this.closeCameraBtn.addEventListener('click', () => this.closeTrainingCamera());
    this.trainBtn.addEventListener('click', () => this.trainModel());
    this.clearBtn.addEventListener('click', () => this.clearTrainingData());
    this.testBtn.addEventListener('click', () => this.startTestMode());
    this.stopTestBtn.addEventListener('click', () => this.stopTestMode());
    
    // Load saved models and update UI
    this.loadSavedModels();
    
    // Update training data display
    this.updateTrainingStats();
    this.updateTrainingDataTable();
    this.trainBtn.disabled = this.trainingData.length === 0;
  }

  async initializeMediaPipe() {
    try {
      // Wait for TensorFlow.js to be loaded
      let retries = 0;
      const maxRetries = 100;
      
      while (retries < maxRetries) {
        if (typeof tf !== 'undefined') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
        retries++;
      }
      
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js is not loaded');
      }

      this.statusText.textContent = 'Loading MobileNet for feature extraction...';
      
      // Load MobileNet model - try multiple sources for reliability
      let loaded = false;
      
      // Try option 1: Load MobileNet v2 from TensorFlow.js model repository (most reliable)
      const mobilenetUrls = [
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json',
        'https://tfhub.dev/tensorflow/tfjs-model/mobilenet_v2_1.0_224/1/default/1',
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1'
      ];
      
      for (const modelUrl of mobilenetUrls) {
        if (loaded) break;
        try {
          this.mobilenet = await tf.loadLayersModel(modelUrl);
          loaded = true;
          console.log('MobileNet v2 loaded successfully from:', modelUrl);
          this.statusText.textContent = 'MobileNet v2 initialized. Ready for training and testing.';
          break;
        } catch (error) {
          console.warn('Failed to load MobileNet from:', modelUrl, error);
        }
      }
      
      // Try option 2: Use official MobileNet package via dynamic import
      if (!loaded) {
        try {
          // Dynamically import MobileNet package from CDN
          const mobilenetModule = await import('https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0/+esm');
          const mobilenet = mobilenetModule.mobilenet || mobilenetModule.default || mobilenetModule;
          
          if (mobilenet && mobilenet.load) {
            // Load MobileNet v2 with 1.0 alpha and 224x224 input size
            this.mobilenet = await mobilenet.load({
              version: 2,
              alpha: 1.0,
              inputSize: 224
            });
            loaded = true;
            console.log('MobileNet v2 loaded successfully from @tensorflow-models/mobilenet package');
            this.statusText.textContent = 'MobileNet v2 initialized. Ready for training and testing.';
          }
        } catch (error) {
          console.warn('Failed to load MobileNet from package:', error);
        }
      }
      
      // Try option 3: TensorFlow Hub as additional fallback
      if (!loaded) {
        const tfHubUrls = [
          'https://tfhub.dev/tensorflow/tfjs-model/mobilenet_v2_1.0_224/1/default/1',
          'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1'
        ];
        
        for (const modelUrl of tfHubUrls) {
          if (loaded) break;
          try {
            this.mobilenet = await tf.loadLayersModel(modelUrl);
            loaded = true;
            console.log('MobileNet loaded successfully from TensorFlow Hub:', modelUrl);
            this.statusText.textContent = 'MobileNet initialized from TensorFlow Hub. Ready for training and testing.';
          } catch (error) {
            console.warn('TensorFlow Hub URL failed:', modelUrl, error);
          }
        }
      }
      
      // Try option 3: Create a lightweight convolutional feature extractor
      if (!loaded) {
        try {
          // Create a simple convolutional feature extractor using standard conv2d layers
          // This is a lightweight alternative that works offline
          this.mobilenet = tf.sequential({
            layers: [
              // First conv block - reduce spatial dimensions
              tf.layers.conv2d({
                inputShape: [224, 224, 3],
                filters: 32,
                kernelSize: 3,
                strides: 2,
                padding: 'same',
                activation: 'relu',
                useBias: true
              }),
              tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
              
              // Second conv block
              tf.layers.conv2d({
                filters: 64,
                kernelSize: 3,
                strides: 1,
                padding: 'same',
                activation: 'relu',
                useBias: true
              }),
              tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
              
              // Third conv block
              tf.layers.conv2d({
                filters: 128,
                kernelSize: 3,
                strides: 1,
                padding: 'same',
                activation: 'relu',
                useBias: true
              }),
              tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
              
              // Fourth conv block
              tf.layers.conv2d({
                filters: 128,
                kernelSize: 3,
                strides: 1,
                padding: 'same',
                activation: 'relu',
                useBias: true
              }),
              tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
              
              // Global average pooling to reduce to feature vector
              tf.layers.globalAveragePooling2d(),
              
              // Dense layer for feature extraction
              tf.layers.dense({
                units: 128,
                activation: 'relu',
                useBias: true
              })
            ]
          });
          
          // Compile the model (needed for proper initialization)
          this.mobilenet.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
          });
          
          // Test the model with a dummy input to ensure it's properly initialized
          const testInput = tf.zeros([1, 224, 224, 3]);
          const testOutput = this.mobilenet.predict(testInput);
          await testOutput.data(); // Wait for prediction to complete
          testInput.dispose();
          testOutput.dispose();
          
          loaded = true;
          console.log('Created lightweight convolutional feature extractor');
          this.statusText.textContent = 'Using lightweight feature extractor (pre-trained MobileNet unavailable). Ready for training and testing.';
        } catch (error) {
          console.warn('Failed to create feature extractor:', error);
          console.error('Feature extractor error details:', error.message, error.stack);
        }
      }
      
      if (!loaded) {
        // Last resort: simplified feature extraction (no MobileNet)
        this.statusText.textContent = 'Error: Could not load MobileNet. Please check your internet connection and refresh the page.';
        this.mobilenet = null;
        console.error('All MobileNet loading methods failed. Using fallback feature extraction method.');
        alert('Warning: MobileNet could not be loaded. The detector will use pixel features which may not work well. Please check your internet connection and refresh the page.');
      }
      
      this.mediaPipeReady = true;
    } catch (error) {
      console.error('Error initializing MobileNet:', error);
      this.statusText.textContent = `Error: ${error.message}. Check console (F12) for details.`;
      this.mobilenet = null;
    }
  }

  // Feature extraction using MobileNet or fallback method
  async extractFeatures(image) {
    if (this.mobilenet) {
      // Check if this is the official MobileNet model (has infer method)
      if (this.mobilenet.infer && typeof this.mobilenet.infer === 'function') {
        // Use official MobileNet API - infer method extracts features
        // The infer method automatically handles preprocessing (resize, normalize)
        const features = this.mobilenet.infer(image, true); // true = embedding (feature vector), false = predictions
        
        // Get feature vector as array
        const featureArray = await features.data();
        
        // Clean up tensors
        features.dispose();
        
        return Array.from(featureArray);
      } else {
        // Use MobileNet loaded as a layers model (from TensorFlow Hub or custom)
        // Preprocess image: resize to 224x224 (MobileNet input size)
        // MobileNet expects images normalized to [-1, 1] range, not [0, 1]
        const tensor = tf.browser.fromPixels(image)
          .resizeNearestNeighbor([224, 224])
          .expandDims(0)
          .toFloat()
          .div(tf.scalar(127.5))
          .sub(tf.scalar(1.0)); // Normalize to [-1, 1]
        
        // For classification models, we need to extract features from before the final layer
        // Create a feature extraction model by removing the last classification layer
        let featureModel = this.mobilenet;
        
        // If this is a classification model (has many output classes), extract from intermediate layer
        if (this.mobilenet.layers && this.mobilenet.layers.length > 0) {
          // Check if last layer is a dense layer with many outputs (classification layer)
          const lastLayer = this.mobilenet.layers[this.mobilenet.layers.length - 1];
          if (lastLayer && lastLayer.outputShape && lastLayer.outputShape[lastLayer.outputShape.length - 1] > 100) {
            // This is likely a classification model, create a feature extraction model
            // by using all layers except the last one
            try {
              const layerOutput = this.mobilenet.layers[this.mobilenet.layers.length - 2].output;
              featureModel = tf.model({
                inputs: this.mobilenet.input,
                outputs: layerOutput
              });
            } catch (error) {
              // If we can't create a feature model, use the full model output
              // (will be classification probabilities, but can still work as features)
              console.warn('Could not extract intermediate features, using full model output:', error);
            }
          }
        }
        
        // Extract features
        let features;
        const predictResult = featureModel.predict(tensor);
        if (predictResult instanceof Promise) {
          features = await predictResult;
        } else {
          features = predictResult;
        }
        
        // Ensure features is a tensor
        if (!(features instanceof tf.Tensor)) {
          features = tf.tensor(features);
        }
        
        // Flatten if needed (in case output is multi-dimensional)
        if (features.shape.length > 2) {
          features = features.flatten();
        }
        
        // Get feature vector as array
        const featureArray = await features.data();
        
        // Clean up tensors
        tensor.dispose();
        features.dispose();
        if (featureModel !== this.mobilenet) {
          featureModel.dispose();
        }
        
        return Array.from(featureArray);
      }
    } else {
      // Fallback: Use simplified feature extraction (should not happen if MobileNet loads correctly)
      console.warn('MobileNet not available, using pixel features (not recommended)');
      // Resize image and extract pixel features
      const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([64, 64]) // Smaller size for efficiency
        .expandDims(0)
        .div(255.0); // Normalize to [0, 1]
      
      // Flatten to get feature vector
      const flattened = tensor.flatten();
      const featureArray = await flattened.data();
      
      // Clean up
      tensor.dispose();
      flattened.dispose();
      
      return Array.from(featureArray);
    }
  }

  // Load image from file
  loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  loadTrainingData() {
    try {
      const saved = localStorage.getItem('animalTrainingData');
      if (saved) {
        const data = JSON.parse(saved);
        this.trainingData = data.trainingData || [];
        console.log(`Loaded ${this.trainingData.length} training samples from storage`);
      }
    } catch (error) {
      console.error('Error loading training data:', error);
      this.trainingData = [];
    }
  }

  async handleFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const animalName = this.animalTypeInput.value.trim();
    if (!animalName) {
      alert('Please enter an animal name first.');
      this.animalTypeInput.focus();
      return;
    }

    if (!this.mediaPipeReady) {
      alert('Feature extractor not ready. Please wait...');
      return;
    }

    let processedCount = 0;
    let errorCount = 0;

    this.statusText.textContent = `Processing ${files.length} image(s)...`;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const image = await this.loadImageFromFile(file);
        const features = await this.extractFeatures(image);
        
        // Convert image to data URL for display
        const canvas = document.createElement('canvas');
        canvas.width = image.width || image.naturalWidth;
        canvas.height = image.height || image.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
        
        this.trainingData.push({
          features: features,
          label: animalName,
          imageDataUrl: imageDataUrl,
          timestamp: Date.now()
        });
        processedCount++;
      } catch (error) {
        console.error(`Error processing image ${file.name}:`, error);
        errorCount++;
      }
    }

    // Clear file input
    this.fileInput.value = '';

    // Save to localStorage
    this.saveTrainingData();

    // Update UI
    this.updateTrainingStats();
    this.updateTrainingDataTable();
    this.trainBtn.disabled = this.trainingData.length === 0;

    // Show feedback
    if (processedCount > 0) {
      this.statusText.textContent = `Processed ${processedCount} ${animalName} sample(s) (Total: ${this.trainingData.length})`;
      if (errorCount > 0) {
        this.statusText.textContent += `. ${errorCount} image(s) had errors.`;
      }
    } else {
      this.statusText.textContent = `Error processing images. Please try again.`;
    }
  }

  async openTrainingCamera() {
    if (this.isTrainingCameraOpen) {
      return; // Already open
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      });
      
      this.trainingStream = stream;
      this.trainingVideo.srcObject = stream;
      this.trainingCameraContainer.style.display = 'block';
      this.openCameraBtn.style.display = 'none';
      this.takeScreenshotBtn.style.display = 'inline-block';
      this.closeCameraBtn.style.display = 'inline-block';
      this.isTrainingCameraOpen = true;
      
      this.statusText.textContent = 'Camera opened. Enter animal name and take screenshots.';
    } catch (error) {
      console.error('Error opening camera:', error);
      alert('Error accessing camera: ' + error.message);
      this.statusText.textContent = 'Error accessing camera';
    }
  }

  async takeScreenshot() {
    if (!this.isTrainingCameraOpen || !this.trainingVideo) {
      alert('Camera not open');
      return;
    }

    if (!this.mediaPipeReady) {
      alert('Feature extractor not ready. Please wait...');
      return;
    }

    const animalName = this.animalTypeInput.value.trim();
    if (!animalName) {
      alert('Please enter an animal name first.');
      this.animalTypeInput.focus();
      return;
    }

    try {
      this.statusText.textContent = 'Processing screenshot...';
      
      // Check if video is ready
      if (!this.trainingVideo || this.trainingVideo.readyState < 2) {
        alert('Camera not ready. Please wait a moment and try again.');
        this.statusText.textContent = 'Camera not ready';
        return;
      }
      
      const videoWidth = this.trainingVideo.videoWidth;
      const videoHeight = this.trainingVideo.videoHeight;
      
      if (!videoWidth || !videoHeight || videoWidth === 0 || videoHeight === 0) {
        alert('Camera video dimensions not available. Please wait a moment and try again.');
        this.statusText.textContent = 'Camera not ready - invalid dimensions';
        return;
      }
      
      // Wait a moment to ensure video frame is stable
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Create a canvas to capture the current frame
      const canvas = document.createElement('canvas');
      canvas.width = videoWidth;
      canvas.height = videoHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      // Mirror the canvas to match what the user sees
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(this.trainingVideo, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      
      // Create image from canvas
      const image = new Image();
      image.src = canvas.toDataURL('image/png');
      
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        setTimeout(() => reject(new Error('Image load timeout')), 3000);
      });
      
      // Extract features
      const features = await this.extractFeatures(image);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      
      this.trainingData.push({
        features: features,
        label: animalName,
        imageDataUrl: imageDataUrl,
        timestamp: Date.now()
      });
      
      // Save to localStorage
      this.saveTrainingData();
      
      // Update UI
      this.updateTrainingStats();
      this.updateTrainingDataTable();
      this.trainBtn.disabled = this.trainingData.length === 0;
      
      // Show feedback
      this.statusText.textContent = `Captured ${animalName} sample from screenshot (Total: ${this.trainingData.length})`;
      
      console.log(`Captured training sample: ${animalName}`, features);
    } catch (error) {
      console.error('Error taking screenshot:', error);
      alert(`Error processing screenshot: ${error.message}`);
      this.statusText.textContent = 'Error processing screenshot';
    }
  }

  closeTrainingCamera() {
    if (this.trainingStream) {
      this.trainingStream.getTracks().forEach(track => track.stop());
      this.trainingStream = null;
    }
    
    if (this.trainingVideo) {
      this.trainingVideo.srcObject = null;
    }
    
    this.trainingCameraContainer.style.display = 'none';
    this.openCameraBtn.style.display = 'inline-block';
    this.takeScreenshotBtn.style.display = 'none';
    this.closeCameraBtn.style.display = 'none';
    this.isTrainingCameraOpen = false;
    
    this.statusText.textContent = 'Camera closed';
  }

  async trainModel() {
    if (this.trainingData.length === 0) {
      alert('Please add training data before training a model.');
      return;
    }

    if (typeof tf === 'undefined') {
      alert('TensorFlow.js is not loaded. Please refresh the page and try again.');
      return;
    }

    // Get model name
    const modelName = this.modelNameInput.value.trim();
    if (!modelName) {
      alert('Please enter a model name before training.');
      this.modelNameInput.focus();
      return;
    }

    // Check if model name already exists
    const existingModels = this.loadSavedModelsList();
    if (existingModels.some(m => m.name.toLowerCase() === modelName.toLowerCase())) {
      if (!confirm(`A model named "${modelName}" already exists. Do you want to overwrite it?`)) {
        return;
      }
      // Delete existing model with same name
      const existingModel = existingModels.find(m => m.name.toLowerCase() === modelName.toLowerCase());
      if (existingModel) {
        this.deleteModel(existingModel.id);
      }
    }

    this.statusText.textContent = 'Training model...';
    this.trainBtn.disabled = true;

    try {
      // Get unique labels from training data and sort them consistently
      const uniqueLabels = [...new Set(this.trainingData.map(sample => sample.label))].sort();
      const numClasses = uniqueLabels.length;
      
      // Create dynamic label mapping
      const labelMap = {};
      uniqueLabels.forEach((label, index) => {
        labelMap[label] = index;
      });
      
      console.log('=== Training Model ===');
      console.log('Unique labels found:', uniqueLabels);
      console.log('Number of classes:', numClasses);
      console.log('Label mapping:', labelMap);
      
      // Prepare training data
      const features = this.trainingData.map(sample => sample.features);
      const labels = this.trainingData.map(sample => {
        return labelMap[sample.label];
      });

      // Convert to tensors
      const xs = tf.tensor2d(features);
      
      // Handle single-class vs multi-class models
      let ys;
      let model;
      let compileConfig;
      
      if (numClasses === 1) {
        // Single-class model: use binary classification
        ys = tf.ones([labels.length, 1]);
        
        model = tf.sequential({
          layers: [
            tf.layers.dense({
              inputShape: [features[0].length],
              units: 64,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 32,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 1,
              activation: 'sigmoid'
            })
          ]
        });

        compileConfig = {
          optimizer: 'adam',
          loss: 'binaryCrossentropy',
          metrics: ['accuracy']
        };
      } else {
        // Multi-class model: use one-hot encoding and softmax
        ys = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
        
        model = tf.sequential({
          layers: [
            tf.layers.dense({
              inputShape: [features[0].length],
              units: 64,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 32,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: numClasses,
              activation: 'softmax'
            })
          ]
        });

        compileConfig = {
          optimizer: 'adam',
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy']
        };
      }
      
      // Compile the model
      model.compile(compileConfig);

      // Train model
      await model.fit(xs, ys, {
        epochs: 100,
        batchSize: Math.min(32, this.trainingData.length),
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 20 === 0) {
              console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
            }
          }
        }
      });

      // Save model
      let modelInfo;
      try {
        modelInfo = await this.saveModel(model, modelName, this.trainingData, uniqueLabels, labelMap);
      } catch (error) {
        if (error.name === 'QuotaExceededError') {
          alert('Storage quota exceeded! Please delete some old models or clear your browser storage. The model was trained but could not be saved.');
          this.statusText.textContent = `Model "${modelName}" trained but could not be saved due to storage limit.`;
          this.trainBtn.disabled = false;
          xs.dispose();
          ys.dispose();
          return;
        } else {
          throw error;
        }
      }
      
      // Clean up tensors
      xs.dispose();
      ys.dispose();

      // Update UI
      this.updateTrainingStats();
      this.statusText.textContent = `Model "${modelName}" trained and saved successfully with ${this.trainingData.length} samples!`;
      
      // Clear model name input
      this.modelNameInput.value = '';
      
      // Optionally clear training data
      if (confirm('Model saved! Do you want to clear the training data to start fresh?')) {
        this.trainingData = [];
        localStorage.removeItem('animalTrainingData');
        this.updateTrainingStats();
        this.trainBtn.disabled = true;
      }
      
      console.log('Model training completed:', modelInfo);
    } catch (error) {
      console.error('Error training model:', error);
      this.statusText.textContent = `Training error: ${error.message}`;
      alert(`Error training model: ${error.message}`);
    } finally {
      this.trainBtn.disabled = false;
    }
  }

  clearTrainingData() {
    if (confirm('Are you sure you want to clear all training data? This cannot be undone.')) {
      this.trainingData = [];
      localStorage.removeItem('animalTrainingData');
      
      // Update UI
      this.updateTrainingStats();
      this.updateTrainingDataTable();
      this.trainBtn.disabled = true;
      this.statusText.textContent = 'Training data cleared';
      
      // Close training camera if open
      if (this.isTrainingCameraOpen) {
        this.closeTrainingCamera();
      }
    }
  }

  async startTestMode() {
    if (!this.trainedModel || !this.selectedModelInfo) {
      alert('Please select a model first.');
      return;
    }

    if (this.isTestModeActive) {
      return; // Already active
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      });
      
      this.testStream = stream;
      this.testVideo.srcObject = stream;
      this.testCameraContainer.style.display = 'block';
      this.testBtn.style.display = 'none';
      this.stopTestBtn.style.display = 'inline-block';
      this.isTestModeActive = true;
      
      // Start detection loop
      this.detectAnimalsInTestMode();
      
      this.statusText.textContent = 'Test mode active. Showing animal detection results.';
    } catch (error) {
      console.error('Error starting test mode:', error);
      alert('Error accessing camera: ' + error.message);
      this.statusText.textContent = 'Error accessing camera';
    }
  }

  stopTestMode() {
    if (this.testAnimationFrame) {
      cancelAnimationFrame(this.testAnimationFrame);
      this.testAnimationFrame = null;
    }
    
    if (this.testStream) {
      this.testStream.getTracks().forEach(track => track.stop());
      this.testStream = null;
    }
    
    if (this.testVideo) {
      this.testVideo.srcObject = null;
    }
    
    this.testCameraContainer.style.display = 'none';
    this.testBtn.style.display = 'inline-block';
    this.stopTestBtn.style.display = 'none';
    this.isTestModeActive = false;
    
    // Reset display
    this.updateTestAnimalDisplay(null, null);
    this.statusText.textContent = 'Test mode stopped';
  }

  async detectAnimalsInTestMode() {
    if (!this.isTestModeActive || !this.testVideo || !this.trainedModel || !this.mediaPipeReady) {
      return;
    }

    if (this.testVideo.readyState === this.testVideo.HAVE_ENOUGH_DATA) {
      // Extract features from current frame
      try {
        const features = await this.extractFeatures(this.testVideo);
        const featuresTensor = tf.tensor2d([features]);
        
        // Predict
        const prediction = this.trainedModel.predict(featuresTensor);
        const probabilities = await prediction.data();
        
        // Clean up
        featuresTensor.dispose();
        prediction.dispose();
        
        // Map index to label
        const trainedLabels = this.selectedModelInfo.trainedLabels || [];
        const labelMap = this.selectedModelInfo.labelMap || {};
        
        let predictedAnimal = null;
        let confidence = 0;
        
        // Handle both binary classification (single class) and multi-class models
        const probArray = Array.from(probabilities);
        
        // Debug logging (throttled to avoid console spam)
        if (!this._lastDebugLogTime || (Date.now() - this._lastDebugLogTime) > 2000) {
          console.log('=== Prediction Debug ===');
          console.log('Probabilities array:', probArray);
          console.log('Array length:', probArray.length);
          console.log('Trained labels:', trainedLabels);
          console.log('Confidence threshold:', this.confidenceThreshold);
          this._lastDebugLogTime = Date.now();
        }
        
        if (probArray.length === 1) {
          // Binary classification model (single class)
          // Output is a single probability value
          confidence = probArray[0];
          
          // For binary classification, check if probability is far from 0.5 (ambiguous)
          // If it's close to 0.5, it's uncertain
          const ambiguityThreshold = 0.3; // If confidence is between 0.2 and 0.8, it's ambiguous
          const isAmbiguous = Math.abs(confidence - 0.5) < ambiguityThreshold;
          
          // For binary classification, if probability is above threshold, it's the trained class
          // UNLESS showRawPredictions is enabled (for debugging)
          if ((this.showRawPredictions || (confidence >= this.confidenceThreshold && !isAmbiguous)) && trainedLabels.length > 0) {
            predictedAnimal = trainedLabels[0];
            // If showing raw prediction but below threshold, add a warning indicator
            if (this.showRawPredictions && (confidence < this.confidenceThreshold || isAmbiguous)) {
              predictedAnimal = trainedLabels[0] + ' (LOW)';
            }
          } else {
            predictedAnimal = 'NONE';
            // Debug: Log why it was rejected
            if (!this._lastDebugLogTime || (Date.now() - this._lastDebugLogTime) > 2000) {
              console.log(`Binary prediction rejected: confidence ${confidence.toFixed(4)} < threshold ${this.confidenceThreshold} or ambiguous (${isAmbiguous})`);
              console.log('Enable "Show raw predictions" checkbox to see the actual prediction anyway.');
            }
          }
        } else {
          // Multi-class model
          // Get predicted class with highest probability
          const sortedProbs = [...probArray].map((p, i) => ({ prob: p, index: i })).sort((a, b) => b.prob - a.prob);
          const maxIndex = sortedProbs[0].index;
          confidence = sortedProbs[0].prob;
          
          // Calculate entropy to detect ambiguous predictions
          // High entropy = probabilities are similar = uncertain prediction
          let entropy = 0;
          for (const p of probArray) {
            if (p > 0.0001) { // Avoid log(0)
              entropy -= p * Math.log2(p);
            }
          }
          const maxEntropy = Math.log2(probArray.length); // Maximum possible entropy
          const normalizedEntropy = entropy / maxEntropy; // 0 = certain, 1 = completely uncertain
          
          // Calculate margin: difference between top 2 probabilities
          // Small margin = ambiguous prediction
          const margin = probArray.length > 1 
            ? sortedProbs[0].prob - sortedProbs[1].prob 
            : 1.0;
          
          // Thresholds for detecting ambiguous/background predictions
          const entropyThreshold = 0.7; // If entropy > 70% of max, it's ambiguous
          const marginThreshold = 0.3; // If margin < 30%, top 2 are too close
          
          // Special handling: If model only has 2 classes (e.g., dog/cat) without background class,
          // be more conservative with very high confidence predictions
          // This helps catch false positives on backgrounds
          const hasBackgroundClass = trainedLabels.some(label => 
            label.toLowerCase().includes('background') || 
            label.toLowerCase().includes('none') ||
            label.toLowerCase().includes('empty')
          );
          
          // If no background class and only 2 classes, require higher confidence threshold
          // to reduce false positives on backgrounds
          const effectiveThreshold = (!hasBackgroundClass && probArray.length === 2) 
            ? Math.max(this.confidenceThreshold, 0.75) // At least 75% confidence for 2-class models
            : this.confidenceThreshold;
          
          const isAmbiguous = normalizedEntropy > entropyThreshold || margin < marginThreshold;
          
          // Debug: Log the prediction details
          if (!this._lastDebugLogTime || (Date.now() - this._lastDebugLogTime) > 2000) {
            console.log('Max probability index:', maxIndex);
            console.log('Max probability value:', confidence);
            console.log('Entropy:', entropy.toFixed(4), 'Normalized:', normalizedEntropy.toFixed(4));
            console.log('Margin (top2 diff):', margin.toFixed(4));
            console.log('Is ambiguous:', isAmbiguous);
            console.log('All probabilities:', probArray.map((p, i) => `${trainedLabels[i] || i}: ${p.toFixed(4)}`).join(', '));
          }
          
          // Only output a prediction if:
          // 1. Confidence is above effective threshold (higher for 2-class models without background)
          // 2. Prediction is not ambiguous (low entropy, high margin)
          // This prevents false positives when there's no object or the object doesn't match
          // UNLESS showRawPredictions is enabled (for debugging)
          const shouldShowPrediction = this.showRawPredictions || (confidence >= effectiveThreshold && !isAmbiguous);
          
          if (shouldShowPrediction) {
            if (trainedLabels.length > 0 && maxIndex >= 0 && maxIndex < trainedLabels.length) {
              predictedAnimal = trainedLabels[maxIndex];
              // If showing raw prediction but below threshold or ambiguous, add a warning indicator
              if (this.showRawPredictions && (confidence < this.confidenceThreshold || isAmbiguous)) {
                predictedAnimal = trainedLabels[maxIndex] + ' (LOW/AMBIG)';
              }
            } else {
              predictedAnimal = 'NONE';
            }
          } else {
            // Confidence is too low or prediction is ambiguous - output "NONE"
            predictedAnimal = 'NONE';
            // Debug: Log why it was rejected
            if (!this._lastDebugLogTime || (Date.now() - this._lastDebugLogTime) > 2000) {
              const reasons = [];
              if (confidence < effectiveThreshold) {
                reasons.push(`confidence ${confidence.toFixed(4)} < effective threshold ${effectiveThreshold.toFixed(4)}`);
                if (effectiveThreshold > this.confidenceThreshold) {
                  reasons.push(`(raised from ${this.confidenceThreshold} because model has no background class)`);
                }
              }
              if (isAmbiguous) {
                reasons.push(`ambiguous (entropy: ${normalizedEntropy.toFixed(2)}, margin: ${margin.toFixed(2)})`);
              }
              console.log(`Prediction rejected: ${reasons.join(', ')}`);
              console.log('💡 Tip: Train with "background" images to improve detection accuracy.');
              console.log('Enable "Show raw predictions" checkbox to see the actual prediction anyway.');
            }
          }
        }
        
        // Update display
        this.updateTestAnimalDisplay(predictedAnimal, confidence);
      } catch (error) {
        console.error('Error in detection:', error);
      }
    }
    
    // Continue loop
    this.testAnimationFrame = requestAnimationFrame(() => this.detectAnimalsInTestMode());
  }

  updateTestAnimalDisplay(animal, confidence) {
    // Update animal box
    if (animal && this.testAnimalValue && this.testAnimalDisplay) {
      this.testAnimalValue.textContent = animal.toUpperCase();
      
      // Color coding: green for valid detection, red for "NONE", gray for no detection
      if (animal === 'NONE') {
        this.testAnimalDisplay.style.background = 'rgba(244, 67, 54, 0.9)'; // Red for "NONE"
        this.testAnimalDisplay.style.color = '#ffffff';
      } else {
        this.testAnimalDisplay.style.background = 'rgba(76, 175, 80, 0.9)'; // Green for valid detection
        this.testAnimalDisplay.style.color = '#ffffff';
      }
    } else {
      if (this.testAnimalValue) {
        this.testAnimalValue.textContent = '--';
      }
      if (this.testAnimalDisplay) {
        this.testAnimalDisplay.style.background = 'rgba(0, 0, 0, 0.7)';
        this.testAnimalDisplay.style.color = '#ffffff';
      }
    }
    
    // Update confidence box
    if (confidence !== null && confidence !== undefined && this.testConfidenceValue && this.testConfidenceDisplay) {
      const confidenceDecimal = confidence.toFixed(2);
      this.testConfidenceValue.textContent = confidenceDecimal;
      
      if (confidence > 0.7) {
        this.testConfidenceDisplay.style.background = 'rgba(76, 175, 80, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      } else if (confidence > 0.5) {
        this.testConfidenceDisplay.style.background = 'rgba(255, 152, 0, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      } else {
        this.testConfidenceDisplay.style.background = 'rgba(244, 67, 54, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      }
    } else {
      if (this.testConfidenceValue) {
        this.testConfidenceValue.textContent = '--';
      }
      if (this.testConfidenceDisplay) {
        this.testConfidenceDisplay.style.background = 'rgba(0, 0, 0, 0.7)';
        this.testConfidenceValue.style.color = '#ffffff';
      }
    }
  }

  saveTrainingData() {
    try {
      const dataToSave = {
        trainingData: this.trainingData,
        timestamp: Date.now()
      };
      const jsonString = JSON.stringify(dataToSave);
      const sizeInMB = new Blob([jsonString]).size / (1024 * 1024);
      console.log('Saving training data. Size:', sizeInMB.toFixed(2), 'MB');
      
      if (sizeInMB > 5) {
        console.warn('Warning: Training data is large (', sizeInMB.toFixed(2), 'MB). localStorage has ~5-10MB limit.');
      }
      
      localStorage.setItem('animalTrainingData', jsonString);
      console.log('Training data saved successfully. Sample count:', this.trainingData.length);
    } catch (error) {
      console.error('Error saving training data:', error);
      if (error.name === 'QuotaExceededError') {
        alert('Error: Training data is too large to save. Please reduce the number of samples or clear old data.');
      }
    }
  }

  sanitizeModelName(name) {
    return name.replace(/[^a-zA-Z0-9_-]/g, '_');
  }

  async saveModel(model, modelName, trainingData, uniqueLabels = null, labelMap = null) {
    try {
      const sanitizedName = this.sanitizeModelName(modelName);
      const storageKey = `animal_model_${sanitizedName}_${Date.now()}`;
      
      await model.save('indexeddb://' + storageKey);
      
      if (!uniqueLabels) {
        uniqueLabels = [...new Set(trainingData.map(sample => sample.label))].sort();
      }
      const labelCounts = {};
      trainingData.forEach(sample => {
        labelCounts[sample.label] = (labelCounts[sample.label] || 0) + 1;
      });
      
      if (!labelMap) {
        labelMap = {};
        uniqueLabels.forEach((label, index) => {
          labelMap[label] = index;
        });
      }
      
      const modelInfo = {
        id: Date.now().toString(),
        name: modelName,
        storageKey: storageKey,
        trainingDataCount: trainingData.length,
        trainedLabels: uniqueLabels,
        labelMap: labelMap,
        labelCounts: labelCounts,
        createdAt: new Date().toISOString()
      };
      
      const savedModels = this.loadSavedModelsList();
      savedModels.push(modelInfo);
      
      const jsonString = JSON.stringify(savedModels);
      const sizeInMB = new Blob([jsonString]).size / (1024 * 1024);
      
      if (sizeInMB > 4) {
        console.warn('Warning: Model metadata is large (', sizeInMB.toFixed(2), 'MB). Consider deleting old models.');
      }
      
      try {
        localStorage.setItem('animalSavedModels', jsonString);
      } catch (storageError) {
        if (storageError.name === 'QuotaExceededError') {
          console.warn('Storage quota exceeded. Attempting to free space...');
          const sortedModels = savedModels.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
          const recentModels = sortedModels.slice(0, 10);
          
          const modelsToDelete = sortedModels.slice(10);
          for (const oldModel of modelsToDelete) {
            try {
              const oldStorageKey = oldModel.storageKey || this.sanitizeModelName(oldModel.name);
              await tf.io.removeModel('indexeddb://' + oldStorageKey);
            } catch (e) {
              console.warn('Could not remove old model from IndexedDB:', e);
            }
          }
          
          const reducedJsonString = JSON.stringify(recentModels);
          localStorage.setItem('animalSavedModels', reducedJsonString);
          console.log('Freed space by removing old models. Kept', recentModels.length, 'most recent models.');
        } else {
          throw storageError;
        }
      }
      
      this.loadSavedModels();
      
      return modelInfo;
    } catch (error) {
      console.error('Error saving model:', error);
      throw error;
    }
  }

  loadSavedModelsList() {
    try {
      const saved = localStorage.getItem('animalSavedModels');
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error('Error loading saved models list:', error);
      return [];
    }
  }

  async loadModel(modelId) {
    try {
      const models = this.loadSavedModelsList();
      const modelInfo = models.find(m => m.id === modelId);
      
      if (!modelInfo) {
        throw new Error('Model not found');
      }
      
      const storageKey = modelInfo.storageKey || this.sanitizeModelName(modelInfo.name);
      const model = await tf.loadLayersModel('indexeddb://' + storageKey);
      
      return { model, modelInfo };
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  deleteModel(modelId) {
    try {
      const models = this.loadSavedModelsList();
      const modelInfo = models.find(m => m.id === modelId);
      
      if (!modelInfo) {
        alert('Model not found');
        return;
      }
      
      if (!confirm(`Are you sure you want to delete model "${modelInfo.name}"?`)) {
        return;
      }
      
      const updatedModels = models.filter(m => m.id !== modelId);
      localStorage.setItem('animalSavedModels', JSON.stringify(updatedModels));
      
      try {
        const storageKey = modelInfo.storageKey || this.sanitizeModelName(modelInfo.name);
        tf.io.removeModel('indexeddb://' + storageKey);
      } catch (e) {
        console.warn('Could not remove model from IndexedDB:', e);
      }
      
      if (this.selectedModelId === modelId) {
        this.selectedModelId = null;
        this.trainedModel = null;
        this.selectedModelInfo = null;
        this.testBtn.disabled = true;
        if (this.isTestModeActive) {
          this.stopTestMode();
        }
        this.testStatus.innerHTML = '<strong>Status:</strong> Selected model was deleted. Please select another model.';
      }
      
      this.loadSavedModels();
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Error deleting model: ' + error.message);
    }
  }

  loadSavedModels() {
    this.savedModels = this.loadSavedModelsList();
    this.updateModelsList();
  }

  updateModelsList() {
    if (!this.modelsList) return;
    
    if (this.savedModels.length === 0) {
      this.modelsList.innerHTML = '<p style="color: #666; margin: 0; text-align: center;">No models saved yet. Train a model to get started.</p>';
      return;
    }
    
    const listHTML = this.savedModels.map(model => {
      const isSelected = this.selectedModelId === model.id;
      const date = new Date(model.createdAt).toLocaleString();
      return `
        <div style="padding: 10px; margin-bottom: 8px; border: 2px solid ${isSelected ? '#4CAF50' : '#ddd'}; border-radius: 4px; background: ${isSelected ? '#f0f8f0' : 'white'}; cursor: pointer;" 
             data-model-id="${model.id}">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <strong style="color: #667eea;">${model.name}</strong>
              <div style="font-size: 12px; color: #666; margin-top: 4px;">
                ${model.trainingDataCount} samples | Created: ${date}
              </div>
            </div>
            <div>
              <button class="select-model-btn" data-model-id="${model.id}" style="padding: 5px 10px; margin-right: 5px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                ${isSelected ? '✓ Selected' : 'Select'}
              </button>
              <button class="delete-model-btn" data-model-id="${model.id}" style="padding: 5px 10px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer;">Delete</button>
            </div>
          </div>
        </div>
      `;
    }).join('');
    
    this.modelsList.innerHTML = listHTML;
    
    // Add event listeners
    this.modelsList.querySelectorAll('.select-model-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.selectModel(btn.dataset.modelId);
      });
    });
    
    this.modelsList.querySelectorAll('.delete-model-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteModel(btn.dataset.modelId);
      });
    });
  }

  async selectModel(modelId) {
    try {
      this.testStatus.innerHTML = '<strong>Status:</strong> Loading model...';
      
      const { model, modelInfo } = await this.loadModel(modelId);
      
      this.selectedModelId = modelId;
      this.trainedModel = model;
      this.selectedModelInfo = modelInfo;
      
      if (!this.trainedModel) {
        throw new Error('Model loaded but is null');
      }
      
      this.testBtn.disabled = false;
      this.testStatus.innerHTML = `<strong>Status:</strong> Model "${modelInfo.name}" (${modelInfo.trainingDataCount} samples) loaded and ready. Click "Start Test" to begin testing.`;
      
      this.updateModelsList();
      
      console.log('✓ Model selected and loaded:', modelInfo.name);
    } catch (error) {
      console.error('Error selecting model:', error);
      alert('Error loading model: ' + error.message);
      this.testStatus.innerHTML = '<strong>Status:</strong> Error loading model. Please try again.';
      this.selectedModelId = null;
      this.trainedModel = null;
      this.selectedModelInfo = null;
    }
  }

  updateTrainingStats() {
    if (this.trainingStats) {
      const counts = {};
      
      // Count samples per animal type
      this.trainingData.forEach(sample => {
        const animalType = sample.label || 'Unknown';
        counts[animalType] = (counts[animalType] || 0) + 1;
      });
      
      const total = this.trainingData.length;
      const modelStatus = this.trainedModel ? 'Trained' : 'Not trained';
      
      // Format counts as string (e.g., "cat: 5, dog: 3, bird: 2")
      const countsStr = Object.keys(counts).length > 0
        ? Object.entries(counts)
            .map(([type, count]) => `${type}: ${count}`)
            .join(', ')
        : '0';
      
      this.trainingStats.innerHTML = `
        <strong>Training Data:</strong> ${total} samples 
        ${total > 0 ? `(${countsStr})` : ''} | 
        <strong>Model Status:</strong> ${modelStatus}
      `;
    }
  }

  updateTrainingDataTable() {
    if (!this.trainingDataTable) return;
    
    if (this.trainingData.length === 0) {
      this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">No training data yet. Upload images or take screenshots to get started.</p>';
      return;
    }
    
    const itemsHTML = this.trainingData.map((sample, index) => {
      return `
        <div style="position: relative; border: 2px solid #ddd; border-radius: 4px; overflow: hidden; background: white;">
          <img src="${sample.imageDataUrl}" alt="Training sample" style="width: 100%; height: 120px; object-fit: cover; display: block;">
          <div style="padding: 5px; background: #f9f9f9; text-align: center; font-size: 12px; color: #333;">
            <strong>${sample.label}</strong>
          </div>
          <button class="delete-sample-btn" data-index="${index}" style="position: absolute; top: 5px; right: 5px; background: rgba(244, 67, 54, 0.9); color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 14px; line-height: 1;">×</button>
        </div>
      `;
    }).join('');
    
    this.trainingDataTable.innerHTML = itemsHTML;
    
    // Add delete event listeners
    this.trainingDataTable.querySelectorAll('.delete-sample-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(btn.dataset.index);
        if (confirm('Delete this training sample?')) {
          this.trainingData.splice(index, 1);
          this.saveTrainingData();
          this.updateTrainingStats();
          this.updateTrainingDataTable();
          this.trainBtn.disabled = this.trainingData.length === 0;
          this.statusText.textContent = 'Sample deleted';
        }
      });
    });
  }
}

// Register the custom element
customElements.define('animal-detector', AnimalDetector);

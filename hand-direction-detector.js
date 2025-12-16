/**
 * Hand Direction Detector Web Component
 * Detects hand direction (up, down, left, right) using the index finger
 * Uses MediaPipe Tasks Vision API
 */
class HandDirectionDetector extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.mediaPipeReady = false;
    this.handLandmarker = null; // Will be created when test mode starts
    
    // Training-related properties
    this.trainingData = [];
    this.trainedModel = null;
    this.imageHandLandmarker = null; // Separate landmarker for image processing
    this.trainingVideo = null; // Separate video element for training camera
    this.trainingStream = null; // Stream for training camera
    this.isTrainingCameraOpen = false; // Track if training camera is open
    
    // Model management
    this.savedModels = []; // Array of saved models with metadata
    this.selectedModelId = null; // Currently selected model for testing
    this.selectedModelInfo = null; // Info about the selected model (for verification)
    this.testVideo = null; // Video element for test mode
    this.testStream = null; // Stream for test camera
    this.isTestModeActive = false; // Track if test mode is active
    
    // MediaPipe hand landmarks indices
    this.WRIST = 0;
    this.INDEX_FINGER_MCP = 5;  // Index finger metacarpophalangeal joint
    this.INDEX_FINGER_TIP = 8;  // Index finger tip
    this.MIDDLE_FINGER_MCP = 9;
    
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
        
        .direction-display {
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
        
        .direction-display.up {
          background: rgba(0, 255, 0, 0.8);
        }
        
        .direction-display.down {
          background: rgba(255, 0, 0, 0.8);
        }
        
        .direction-display.left {
          background: rgba(255, 165, 0, 0.8);
        }
        
        .direction-display.right {
          background: rgba(0, 100, 255, 0.8);
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
        
        .training-controls select {
          padding: 8px 12px;
          font-size: 14px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
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
          <div class="training-controls-row">
            <select id="directionSelect">
              <option value="up">UP</option>
              <option value="down">DOWN</option>
              <option value="left">LEFT</option>
              <option value="right">RIGHT</option>
            </select>
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
          <strong>Training Data:</strong> 0 samples (UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0)
        </div>
        <div id="trainingDataList" style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px; max-height: 400px; overflow-y: auto;">
          <h4 style="margin-top: 0; color: #667eea;">Training Data List:</h4>
          <div id="trainingDataTable" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; min-height: 50px;">
            <p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">No training data yet. Upload images or take screenshots to get started.</p>
          </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px;">
          <label for="modelNameInput" style="display: block; margin-bottom: 5px; font-weight: bold; color: #667eea;">Model Name:</label>
          <input type="text" id="modelNameInput" placeholder="Enter model name (e.g., MyHandModel)" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">
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
          <button id="testBtn" class="toggle-btn" disabled>Start Test</button>
          <button id="stopTestBtn" class="clear-btn" style="display: none;">Stop Test</button>
        </div>
        <div id="testCameraContainer" style="display: none; margin-top: 15px; text-align: center;">
          <div style="position: relative; display: inline-block; width: 100%; max-width: 640px;">
            <video id="testVideo" autoplay playsinline style="width: 100%; border-radius: 8px; background: #000; transform: scaleX(-1);"></video>
            <canvas id="testCanvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
          </div>
           <div id="testResultContainer" style="position: relative; margin-top: 10px; display: flex; gap: 10px; justify-content: center;">
             <div id="testDirectionDisplay" style="flex: 1; max-width: 200px; padding: 10px 15px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; background: rgba(0, 0, 0, 0.7); color: #fff; min-height: 45px; display: flex; align-items: center; justify-content: center;">
               <span id="testDirectionValue">--</span>
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
    this.directionSelect = this.shadowRoot.getElementById('directionSelect');
    this.fileInput = this.shadowRoot.getElementById('fileInput');
    this.trainingStats = this.shadowRoot.getElementById('trainingStats');
    this.trainingCameraContainer = this.shadowRoot.getElementById('trainingCameraContainer');
    this.trainingVideo = this.shadowRoot.getElementById('trainingVideo');
    this.modelNameInput = this.shadowRoot.getElementById('modelNameInput');
    this.trainingDataTable = this.shadowRoot.getElementById('trainingDataTable');
    
    // Verify element was found
    if (!this.trainingDataTable) {
      console.error('CRITICAL: trainingDataTable element not found after render!');
    } else {
      console.log('✓ trainingDataTable element found successfully');
    }
    
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
    this.testDirectionDisplay = this.shadowRoot.getElementById('testDirectionDisplay');
    this.testDirectionValue = this.shadowRoot.getElementById('testDirectionValue');
    this.testConfidenceDisplay = this.shadowRoot.getElementById('testConfidenceDisplay');
    this.testConfidenceValue = this.shadowRoot.getElementById('testConfidenceValue');
    this.testStatus = this.shadowRoot.getElementById('testStatus');
    
    // Setup button handlers - use arrow functions to preserve 'this' context
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
    
    // Update training data display (data was loaded in constructor, but elements are now ready)
    this.updateTrainingStats();
    this.updateTrainingDataTable();
    this.trainBtn.disabled = this.trainingData.length === 0;
    
    // Debug: Log to verify data and element
    console.log('After render - Training data count:', this.trainingData.length);
    console.log('Training data table element exists:', !!this.trainingDataTable);
    if (this.trainingData.length > 0) {
      console.log('First sample:', this.trainingData[0]);
    }
  }

  async initializeMediaPipe() {
    try {
      // Wait for MediaPipe to be loaded
      let retries = 0;
      const maxRetries = 100;
      
      while (retries < maxRetries) {
        if (window.mediaPipeReady && window.MediaPipeFilesetResolver && window.MediaPipeHandLandmarker) {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
        retries++;
      }
      
      if (!window.MediaPipeFilesetResolver || !window.MediaPipeHandLandmarker) {
        throw new Error('MediaPipe libraries not loaded');
      }

      this.statusText.textContent = 'Initializing MediaPipe...';

      const FilesetResolver = window.MediaPipeFilesetResolver;
      const HandLandmarker = window.MediaPipeHandLandmarker;

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );

      // Create landmarker for image processing (used in training and test mode)
      // Lower confidence thresholds for better detection in training images
      this.imageHandLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "CPU"
        },
        numHands: 1,
        runningMode: "IMAGE",
        minHandDetectionConfidence: 0.3,  // Lowered from 0.5 for better detection
        minHandPresenceConfidence: 0.3,    // Lowered from 0.5 for better detection
        minTrackingConfidence: 0.3         // Lowered from 0.5 for better detection
      });

      this.mediaPipeReady = true;
      this.statusText.textContent = 'MediaPipe initialized. Ready for training and testing.';
    } catch (error) {
      console.error('Error initializing MediaPipe:', error);
      this.statusText.textContent = `Error: ${error.message}. Check console (F12) for details.`;
    }
  }

  detectDirection(landmarks) {
    const indexTip = landmarks[this.INDEX_FINGER_TIP];
    const indexMCP = landmarks[this.INDEX_FINGER_MCP];
    
    // Calculate vector from index finger MCP (base) to tip (pointing direction)
    // This gives us the actual direction the finger is pointing
    // Note: In MediaPipe coordinates, y increases downward (0=top, 1=bottom)
    const dx = indexTip.x - indexMCP.x;
    const dy = indexTip.y - indexMCP.y;
    
    // Calculate distance to determine if finger is extended
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Check if index finger is extended enough
    if (distance < 0.08) {
      return null; // Finger not extended enough
    }
    
    // IMPORTANT: MediaPipe processes the original (non-mirrored) video stream
    // The video display is mirrored (scaleX(-1)) but MediaPipe sees the original
    // So we need to flip x coordinate to account for mirrored display
    // This way, when user points left, we detect left correctly
    
    // Flip x coordinate to account for mirrored display
    const userDx = -dx;  // Flip for user's perspective
    const userDy = dy;   // Y doesn't need flipping (up/down are the same)
    
    // Use angle-based detection for more accurate direction
    // Math.atan2(dy, dx) returns:
    // - Right (dx > 0, dy = 0): 0°
    // - Down (dx = 0, dy > 0): 90°
    // - Left (dx < 0, dy = 0): 180° or -180°
    // - Up (dx = 0, dy < 0): -90° or 270°
    const angle = Math.atan2(userDy, userDx) * (180 / Math.PI);
    
    // Normalize angle to 0-360 range
    let normalizedAngle = angle;
    if (normalizedAngle < 0) {
      normalizedAngle += 360;
    }
    
    // Map angle to direction with 45° tolerance zones
    // Right: -45° to 45° (or 315° to 360° and 0° to 45°)
    // Down: 45° to 135°
    // Left: 135° to 225°
    // Up: 225° to 315°
    
    if ((normalizedAngle >= 0 && normalizedAngle < 45) || 
        (normalizedAngle >= 315 && normalizedAngle <= 360)) {
      return 'right';
    } else if (normalizedAngle >= 45 && normalizedAngle < 135) {
      return 'down';
    } else if (normalizedAngle >= 135 && normalizedAngle < 225) {
      return 'left';
    } else if (normalizedAngle >= 225 && normalizedAngle < 315) {
      return 'up';
    }
    
    // Fallback (shouldn't reach here)
    return null;
  }

  // Training methods
  async handleFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const direction = this.directionSelect.value;
    let processedCount = 0;
    let errorCount = 0;

    this.statusText.textContent = `Processing ${files.length} image(s)...`;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const image = await this.loadImageFromFile(file);
        const landmarks = await this.detectHandsInImage(image);
        
        if (landmarks && landmarks.length > 0) {
          const features = this.extractFeatures(landmarks[0]);
          
          // Convert image to data URL for display
          const canvas = document.createElement('canvas');
          canvas.width = image.width || image.naturalWidth;
          canvas.height = image.height || image.naturalHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(image, 0, 0);
          const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
          
          this.trainingData.push({
            features: features,
            label: direction,
            imageDataUrl: imageDataUrl,
            timestamp: Date.now()
          });
          processedCount++;
        } else {
          errorCount++;
          console.warn(`No hand detected in image: ${file.name}`);
        }
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
      this.statusText.textContent = `Processed ${processedCount} ${direction.toUpperCase()} sample(s) (Total: ${this.trainingData.length})`;
      if (errorCount > 0) {
        this.statusText.textContent += `. ${errorCount} image(s) had no hand detected.`;
      }
    } else {
      this.statusText.textContent = `No hands detected in uploaded images. Please try different images.`;
    }
  }

  async openTrainingCamera() {
    if (this.isTrainingCameraOpen) {
      return; // Already open
    }

    try {
      if (!this.mediaPipeReady || !this.imageHandLandmarker) {
        alert('MediaPipe not initialized yet. Please wait...');
        return;
      }

      // Request camera access
      this.trainingStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      this.trainingVideo.srcObject = this.trainingStream;
      await this.trainingVideo.play();
      
      // Wait for video to be ready with actual dimensions
      await new Promise((resolve) => {
        const checkReady = () => {
          if (this.trainingVideo.readyState >= 2 && this.trainingVideo.videoWidth > 0) {
            console.log('Training video ready:', {
              width: this.trainingVideo.videoWidth,
              height: this.trainingVideo.videoHeight,
              readyState: this.trainingVideo.readyState
            });
            resolve();
          } else {
            setTimeout(checkReady, 100);
          }
        };
        checkReady();
      });
      
      // Show training camera and buttons
      this.trainingCameraContainer.style.display = 'block';
      this.openCameraBtn.style.display = 'none';
      this.takeScreenshotBtn.style.display = 'inline-block';
      this.closeCameraBtn.style.display = 'inline-block';
      
      this.isTrainingCameraOpen = true;
      this.statusText.textContent = 'Training camera ready. Position your hand and click "Take Screenshot"';
    } catch (error) {
      console.error('Error opening training camera:', error);
      alert(`Error opening camera: ${error.message}. Please check permissions.`);
    }
  }

  closeTrainingCamera() {
    if (this.trainingStream) {
      const tracks = this.trainingStream.getTracks();
      tracks.forEach(track => track.stop());
      this.trainingStream = null;
    }
    
    if (this.trainingVideo) {
      this.trainingVideo.srcObject = null;
    }
    
    // Hide training camera and buttons
    this.trainingCameraContainer.style.display = 'none';
    this.openCameraBtn.style.display = 'inline-block';
    this.takeScreenshotBtn.style.display = 'none';
    this.closeCameraBtn.style.display = 'none';
    
    this.isTrainingCameraOpen = false;
    this.statusText.textContent = 'Training camera closed';
  }

  async takeScreenshot() {
    if (!this.isTrainingCameraOpen || !this.trainingVideo) {
      alert('Please open the training camera first.');
      return;
    }

    if (!this.imageHandLandmarker) {
      alert('MediaPipe not initialized. Please wait...');
      return;
    }

    const direction = this.directionSelect.value;
    
    if (!direction) {
      alert('Please select a direction first.');
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
      
      // Ensure video has valid dimensions
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
      
      // Mirror the canvas to match what the user sees in the video (which is mirrored)
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      
      // Draw the video frame to canvas - ensure we capture the current frame
      ctx.drawImage(this.trainingVideo, 0, 0, canvas.width, canvas.height);
      
      // Restore context
      ctx.restore();
      
      console.log('Screenshot captured:', {
        width: canvas.width,
        height: canvas.height,
        videoWidth: videoWidth,
        videoHeight: videoHeight
      });
      
      // Try multiple detection methods in sequence
      let landmarks = null;
      let detectionResults = null;
      let lastError = null;
      
      // Method 1: Try with HTMLImageElement (most reliable for MediaPipe IMAGE mode)
      try {
        console.log('Method 1: Trying HTMLImageElement...');
        const image = new Image();
        image.crossOrigin = 'anonymous';
        
        // Convert canvas to data URL
        const dataUrl = canvas.toDataURL('image/png');
        
        // Wait for image to load
        await new Promise((resolve, reject) => {
          image.onload = () => {
            console.log('Image loaded successfully, dimensions:', image.width, 'x', image.height);
            resolve();
          };
          image.onerror = (e) => {
            console.error('Image load error:', e);
            reject(new Error('Image failed to load'));
          };
          // Set timeout
          setTimeout(() => {
            if (!image.complete) {
              reject(new Error('Image load timeout'));
            }
          }, 3000);
          image.src = dataUrl;
        });
        
        // Detect with loaded image
        detectionResults = this.imageHandLandmarker.detect(image);
        landmarks = detectionResults?.landmarks || [];
        
        if (landmarks.length > 0) {
          console.log('✓ Method 1 SUCCESS: Found', landmarks.length, 'hand(s)');
        } else {
          console.log('Method 1: No hands found, trying next method...');
        }
      } catch (error) {
        console.warn('Method 1 failed:', error.message);
        lastError = error;
      }
      
      // Method 2: Try with canvas element directly
      if (!landmarks || landmarks.length === 0) {
        try {
          console.log('Method 2: Trying HTMLCanvasElement...');
          detectionResults = this.imageHandLandmarker.detect(canvas);
          landmarks = detectionResults?.landmarks || [];
          
          if (landmarks.length > 0) {
            console.log('✓ Method 2 SUCCESS: Found', landmarks.length, 'hand(s)');
          } else {
            console.log('Method 2: No hands found, trying next method...');
          }
        } catch (error) {
          console.warn('Method 2 failed:', error.message);
          lastError = error;
        }
      }
      
      // Method 3: Try with ImageData
      if (!landmarks || landmarks.length === 0) {
        try {
          console.log('Method 3: Trying ImageData...');
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          detectionResults = this.imageHandLandmarker.detect(imageData);
          landmarks = detectionResults?.landmarks || [];
          
          if (landmarks.length > 0) {
            console.log('✓ Method 3 SUCCESS: Found', landmarks.length, 'hand(s)');
          } else {
            console.log('Method 3: No hands found, trying next method...');
          }
        } catch (error) {
          console.warn('Method 3 failed:', error.message);
          lastError = error;
        }
      }
      
      // Method 4: Try with blob URL (sometimes more reliable)
      if (!landmarks || landmarks.length === 0) {
        try {
          console.log('Method 4: Trying Blob URL...');
          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
          const blobUrl = URL.createObjectURL(blob);
          
          const image = new Image();
          image.crossOrigin = 'anonymous';
          
          await new Promise((resolve, reject) => {
            image.onload = resolve;
            image.onerror = reject;
            setTimeout(() => reject(new Error('Blob image load timeout')), 3000);
            image.src = blobUrl;
          });
          
          detectionResults = this.imageHandLandmarker.detect(image);
          landmarks = detectionResults?.landmarks || [];
          
          // Clean up blob URL
          URL.revokeObjectURL(blobUrl);
          
          if (landmarks.length > 0) {
            console.log('✓ Method 4 SUCCESS: Found', landmarks.length, 'hand(s)');
          } else {
            console.log('Method 4: No hands found.');
          }
        } catch (error) {
          console.warn('Method 4 failed:', error.message);
          lastError = error;
        }
      }
      
      // Log final results
      if (landmarks && landmarks.length > 0) {
        console.log('✓✓✓ HAND DETECTED! ✓✓✓');
        console.log('First hand has', landmarks[0].length, 'landmarks');
        console.log('Wrist:', landmarks[0][this.WRIST]);
        console.log('Index MCP:', landmarks[0][this.INDEX_FINGER_MCP]);
        console.log('Index Tip:', landmarks[0][this.INDEX_FINGER_TIP]);
        
        if (detectionResults?.handedness) {
          console.log('Handedness:', detectionResults.handedness);
        }
      } else {
        console.error('✗✗✗ NO HAND DETECTED ✗✗✗');
        console.error('All detection methods failed.');
        console.error('Detection results object:', detectionResults);
        console.error('Last error:', lastError);
        
        // Additional debugging: check if canvas has valid data
        const testImageData = ctx.getImageData(0, 0, Math.min(100, canvas.width), Math.min(100, canvas.height));
        const hasNonZeroPixels = testImageData.data.some((val, idx) => idx % 4 !== 3 && val !== 0);
        console.log('Canvas has image data:', hasNonZeroPixels);
        console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
      }
      
      if (!landmarks || landmarks.length === 0) {
        alert('No hand detected in the screenshot. Please make sure:\n- Your hand is clearly visible\n- Your hand is well-lit\n- Your index finger is extended\n- Try moving closer to the camera');
        this.statusText.textContent = 'No hand detected. Try again.';
        return;
      }
      
      // Extract features
      const features = this.extractFeatures(landmarks[0]);
      
      if (!features || features.length === 0) {
        alert('Failed to extract features from hand. Please try again.');
        return;
      }
      
      // Capture image as data URL for display
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      
      // Add to training data with image
      this.trainingData.push({
        features: features,
        label: direction,
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
      this.statusText.textContent = `Captured ${direction.toUpperCase()} sample from screenshot (Total: ${this.trainingData.length})`;
      
      console.log(`Captured training sample: ${direction}`, features);
    } catch (error) {
      console.error('Error taking screenshot:', error);
      alert(`Error processing screenshot: ${error.message}`);
      this.statusText.textContent = 'Error processing screenshot';
    }
  }

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

  async detectHandsInImage(image) {
    if (!this.imageHandLandmarker) {
      throw new Error('Image hand landmarker not initialized');
    }

    try {
      // MediaPipe HandLandmarker.detect can accept HTMLImageElement, ImageData, or HTMLCanvasElement
      const results = this.imageHandLandmarker.detect(image);
      return results.landmarks || [];
    } catch (error) {
      console.error('Error detecting hands in image:', error);
      // Fallback: try with canvas ImageData
      try {
        let tempCanvas, tempCtx;
        
        // If image is already a canvas, use it directly
        if (image instanceof HTMLCanvasElement) {
          tempCanvas = image;
          tempCtx = tempCanvas.getContext('2d');
        } else {
          // Create canvas from image
          tempCanvas = document.createElement('canvas');
          tempCanvas.width = image.width || image.naturalWidth || 640;
          tempCanvas.height = image.height || image.naturalHeight || 480;
          tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(image, 0, 0);
        }
        
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const results = this.imageHandLandmarker.detect(imageData);
        return results.landmarks || [];
      } catch (err) {
        console.error('Error with ImageData fallback:', err);
        return null;
      }
    }
  }

  extractFeatures(landmarks) {
    // Extract normalized features from hand landmarks
    // Use key points: wrist, index finger MCP, index finger tip
    const wrist = landmarks[this.WRIST];
    const indexMCP = landmarks[this.INDEX_FINGER_MCP];
    const indexTip = landmarks[this.INDEX_FINGER_TIP];
    
    // Calculate relative positions (normalized)
    const dx = indexTip.x - indexMCP.x;
    const dy = indexTip.y - indexMCP.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Normalize by distance to make it scale-invariant
    const normalizedDx = distance > 0 ? dx / distance : 0;
    const normalizedDy = distance > 0 ? dy / distance : 0;
    
    // Also include wrist position relative to finger for context
    const wristDx = wrist.x - indexMCP.x;
    const wristDy = wrist.y - indexMCP.y;
    const wristDist = Math.sqrt(wristDx * wristDx + wristDy * wristDy);
    const normalizedWristDx = wristDist > 0 ? wristDx / wristDist : 0;
    const normalizedWristDy = wristDist > 0 ? wristDy / wristDist : 0;
    
    // Return feature vector (6 features)
    return [
      normalizedDx,
      normalizedDy,
      normalizedWristDx,
      normalizedWristDy,
      distance, // Keep distance for scale information
      Math.atan2(dy, dx) // Angle
    ];
  }

  async trainModel() {
    if (this.trainingData.length === 0) {
      alert('No training data available. Please capture some images first.');
      return;
    }

    // Check how many unique directions are in the training data
    const uniqueLabels = new Set(this.trainingData.map(sample => sample.label));
    const labelCounts = {};
    this.trainingData.forEach(sample => {
      labelCounts[sample.label] = (labelCounts[sample.label] || 0) + 1;
    });
    
    if (uniqueLabels.size === 1) {
      const onlyLabel = Array.from(uniqueLabels)[0];
      if (!confirm(`Warning: You are training with only "${onlyLabel.toUpperCase()}" direction (${this.trainingData.length} samples).\n\nThe model will always predict "${onlyLabel.toUpperCase()}" regardless of the actual finger direction.\n\nFor proper detection, please train with samples from all 4 directions (UP, DOWN, LEFT, RIGHT).\n\nDo you want to continue anyway?`)) {
        return;
      }
    } else if (uniqueLabels.size < 4) {
      const missing = ['up', 'down', 'left', 'right'].filter(d => !uniqueLabels.has(d));
      if (!confirm(`Warning: Your training data is missing ${missing.length} direction(s): ${missing.map(d => d.toUpperCase()).join(', ')}.\n\nCurrent data: ${Object.entries(labelCounts).map(([label, count]) => `${label.toUpperCase()}: ${count}`).join(', ')}\n\nFor best accuracy, train with samples from all 4 directions.\n\nDo you want to continue anyway?`)) {
        return;
      }
    }

    if (this.trainingData.length < 4) {
      alert('Please capture at least 4 samples (one for each direction) for better accuracy.');
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
      
      // Create dynamic label mapping based on actual training data
      // This ensures the model only outputs classes it was trained on
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
      
      // Handle single-class vs multi-class models differently
      let ys;
      let model;
      let compileConfig;
      
      if (numClasses === 1) {
        // Single-class model: use binary classification approach
        // All labels are 0 (or we can use 1, but 0 is more standard for binary)
        // We'll use a single output with sigmoid activation
        ys = tf.ones([labels.length, 1]); // All samples are positive (the single class)
        
        // Create model with single output for binary classification
        model = tf.sequential({
          layers: [
            tf.layers.dense({
              inputShape: [6],
              units: 16,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 8,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 1, // Single output for binary classification
              activation: 'sigmoid'
            })
          ]
        });

        // Compile model with binary crossentropy
        compileConfig = {
          optimizer: 'adam',
          loss: 'binaryCrossentropy',
          metrics: ['accuracy']
        };
      } else {
        // Multi-class model: use one-hot encoding and softmax
        ys = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
        
        // Create model with dynamic number of output classes
        model = tf.sequential({
          layers: [
            tf.layers.dense({
              inputShape: [6],
              units: 16,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: 8,
              activation: 'relu'
            }),
            tf.layers.dense({
              units: numClasses, // Dynamic number of classes
              activation: 'softmax'
            })
          ]
        });

        // Compile model with categorical crossentropy
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

      // Save model with name and label mapping
      let modelInfo;
      try {
        modelInfo = await this.saveModel(model, modelName, this.trainingData, uniqueLabels, labelMap);
      } catch (error) {
        if (error.name === 'QuotaExceededError') {
          alert('Storage quota exceeded! Please delete some old models or clear your browser storage. The model was trained but could not be saved.');
          this.statusText.textContent = `Model "${modelName}" trained but could not be saved due to storage limit.`;
          this.trainBtn.disabled = false;
          // Clean up tensors before returning
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
      
      // Optionally clear training data (user can choose to keep it for another model)
      if (confirm('Model saved! Do you want to clear the training data to start fresh?')) {
        this.trainingData = [];
        localStorage.removeItem('handDirectionTrainingData');
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

  detectDirectionWithTrainedModel(mirroredLandmarks, originalLandmarks = null) {
    // Verify we have a trained model - if not, return null (use ONLY the selected model)
    if (!this.trainedModel) {
      console.warn('WARNING: detectDirectionWithTrainedModel called but no trained model available!');
      return null;
    }

    try {
      // Extract features from mirrored landmarks (model was trained on mirrored images)
      const features = this.extractFeatures(mirroredLandmarks);
      
      // Predict using ONLY the selected trained model
      const prediction = this.trainedModel.predict(tf.tensor2d([features]));
      const probabilities = prediction.dataSync();
      prediction.dispose();
      
      // Check what directions this model was trained on
      // Use trainedLabels from model metadata (we no longer store full trainingData to save space)
      const trainedLabels = this.selectedModelInfo?.trainedLabels || [];
      const labelMap = this.selectedModelInfo?.labelMap || null;
      
      // Debug logging
      if (!this._lastDebugLogTime || (Date.now() - this._lastDebugLogTime) > 5000) {
        console.log('=== Prediction Debug Info ===');
        console.log('Model Info:', this.selectedModelInfo);
        console.log('trainedLabels:', trainedLabels);
        console.log('labelMap:', labelMap);
        console.log('Has labelMap:', !!labelMap);
        this._lastDebugLogTime = Date.now();
      }
      
      let trainedClasses = [];
      let isSingleClassModel = false;
      let isBinaryClassificationModel = false; // New: tracks if model uses binary classification (1 output)
      
      if (trainedLabels.length > 0) {
        // New model format - use trainedLabels directly
        trainedClasses = trainedLabels;
        isSingleClassModel = trainedClasses.length === 1;
        // Check if this is a binary classification model (1 output neuron)
        isBinaryClassificationModel = probabilities.length === 1 && isSingleClassModel;
      } else {
        // Old model format - try to detect single-class by checking label counts or trainingData
        const labelCounts = this.selectedModelInfo?.labelCounts || {};
        const trainingData = this.selectedModelInfo?.trainingData || [];
        
        if (Object.keys(labelCounts).length > 0) {
          // Use labelCounts if available
          const nonZeroLabels = Object.keys(labelCounts).filter(label => labelCounts[label] > 0).sort();
          trainedClasses = nonZeroLabels;
          isSingleClassModel = trainedClasses.length === 1;
          isBinaryClassificationModel = probabilities.length === 1 && isSingleClassModel;
        } else if (trainingData.length > 0) {
          // Fallback to old format with full trainingData
          const uniqueLabels = new Set(trainingData.map(sample => sample.label));
          trainedClasses = Array.from(uniqueLabels).sort();
          isSingleClassModel = trainedClasses.length === 1;
          isBinaryClassificationModel = probabilities.length === 1 && isSingleClassModel;
        } else {
          // Can't determine, assume multi-class with all 4 directions (backward compatibility)
          trainedClasses = ['down', 'left', 'right', 'up'].sort(); // Sort for consistency
          isSingleClassModel = false;
          isBinaryClassificationModel = false;
        }
      }
      
      // Get predicted class using dynamic label mapping
      // Handle binary classification models (1 output) differently from multi-class models
      let maxIndex;
      let predictedClass;
      let confidence;
      
      if (isBinaryClassificationModel) {
        // Binary classification model: single output represents probability of the positive class
        // The output is the probability that the input matches the trained class
        confidence = probabilities[0];
        predictedClass = trainedClasses[0]; // The single trained class
        
        // Enhanced debug logging for binary classification
        if (!this._lastPredictionDebugTime || (Date.now() - this._lastPredictionDebugTime) > 2000) {
          console.log('=== Binary Classification Model Prediction ===');
          console.log('probabilities array:', probabilities);
          console.log('probabilities.length:', probabilities.length);
          console.log('confidence (probability):', confidence);
          console.log('trainedClass:', predictedClass);
          this._lastPredictionDebugTime = Date.now();
        }
      } else {
        // Multi-class model: find the class with highest probability
        maxIndex = probabilities.indexOf(Math.max(...probabilities));
        
        // Enhanced debug logging for prediction
        if (!this._lastPredictionDebugTime || (Date.now() - this._lastPredictionDebugTime) > 2000) {
          console.log('=== Prediction Mapping Debug ===');
          console.log('probabilities array:', probabilities);
          console.log('probabilities.length:', probabilities.length);
          console.log('maxIndex:', maxIndex);
          console.log('max probability:', probabilities[maxIndex]);
          console.log('labelMap exists:', !!labelMap);
          console.log('trainedLabels:', trainedLabels);
          console.log('trainedLabels.length:', trainedLabels.length);
          console.log('trainedClasses:', trainedClasses);
          this._lastPredictionDebugTime = Date.now();
        }
        
        if (labelMap && trainedLabels.length > 0 && trainedLabels.length === probabilities.length) {
          // New format: Map index back to label using trainedLabels array
          // The probabilities array indices correspond to the sorted trainedLabels
          if (maxIndex >= 0 && maxIndex < trainedLabels.length) {
            predictedClass = trainedLabels[maxIndex];
            console.log('New format: mapped index', maxIndex, 'to class', predictedClass);
          } else {
            console.error('Invalid maxIndex:', maxIndex, 'for trainedLabels:', trainedLabels, 'probabilities.length:', probabilities.length);
            return null;
          }
        } else {
          // Old format: Use hardcoded mapping for backward compatibility
          // Old models always have 4 output classes: ['up', 'down', 'left', 'right']
          const oldClasses = ['up', 'down', 'left', 'right'];
          const oldLabelMap = { 'up': 0, 'down': 1, 'left': 2, 'right': 3 };
          
          console.log('Using OLD format mapping');
          console.log('Old classes order:', oldClasses);
          console.log('All probabilities:', oldClasses.map((c, i) => `${c}:${probabilities[i]?.toFixed(3) || 'N/A'}`).join(', '));
          
          // For old models, probabilities array should have 4 elements
          // Map old indices to new class names
          if (maxIndex >= 0 && maxIndex < oldClasses.length && maxIndex < probabilities.length) {
            const oldClass = oldClasses[maxIndex];
            console.log('Old format: index', maxIndex, 'maps to class', oldClass, 'with probability', probabilities[maxIndex]);
            // Check if this old class is in the trained classes
            if (trainedClasses.includes(oldClass)) {
              predictedClass = oldClass;
              console.log('Class', oldClass, 'is in trainedClasses, using it');
            } else {
              // Old model predicted a class not in trained set - invalid
              console.warn('Old model predicted untrained class:', oldClass, 'trainedClasses:', trainedClasses, 'maxIndex:', maxIndex);
              return null;
            }
          } else {
            console.error('Invalid maxIndex for old format:', maxIndex, 'oldClasses.length:', oldClasses.length, 'probabilities.length:', probabilities.length);
            return null;
          }
        }
        
        confidence = probabilities[maxIndex];
      }
      
      // Get geometric detection early so we can use it for validation
      const landmarksForValidation = originalLandmarks || mirroredLandmarks;
      const actualDirection = this.detectDirection(landmarksForValidation);
      
      // Log model information for debugging (only once per second to avoid spam)
      if (!this._lastModelLogTime || (Date.now() - this._lastModelLogTime) > 1000) {
        console.log('=== Model Detection Info ===');
        console.log('Model Name:', this.selectedModelInfo?.name);
        console.log('Trained Classes (directions this model knows):', trainedClasses);
        console.log('Is Single-Class Model:', isSingleClassModel);
        // Map probabilities to class names
        // For new format: use trainedLabels (which matches probability indices)
        // For old format: use oldClasses but only show trained classes
        let probMap;
        if (labelMap && trainedLabels.length > 0) {
          // New format: probabilities array matches trainedLabels array
          probMap = trainedLabels.map((label, i) => `${label}:${probabilities[i].toFixed(3)}`).join(', ');
        } else {
          // Old format: probabilities array has 4 elements for ['up', 'down', 'left', 'right']
          const oldClasses = ['up', 'down', 'left', 'right'];
          probMap = oldClasses.map((label, i) => {
            if (i < probabilities.length) {
              return `${label}:${probabilities[i].toFixed(3)}`;
            }
            return '';
          }).filter(s => s !== '').join(', ');
        }
        console.log('All Probabilities:', probMap);
        console.log('Predicted Class:', predictedClass, '(confidence:', confidence.toFixed(3) + ')');
        console.log('Geometric Detection:', actualDirection);
        this._lastModelLogTime = Date.now();
      }
      
      // If model was trained on only one class, validate the prediction
      // by checking if the actual finger direction matches the trained class
      if (isSingleClassModel) {
        const trainedClass = trainedClasses[0];
        
        // Use default detection to check actual finger direction
        // detectDirection expects non-mirrored landmarks (it does its own flipping)
        // If originalLandmarks provided, use them; otherwise use mirrored (less accurate)
        const landmarksForValidation = originalLandmarks || mirroredLandmarks;
        const actualDirection = this.detectDirection(landmarksForValidation);
        
        // For binary classification models, check confidence threshold first
        if (isBinaryClassificationModel) {
          // Binary classification: confidence is the probability of matching the trained class
          // Only accept if confidence is above threshold (e.g., 0.5)
          if (confidence < 0.5) {
            // Low confidence - model doesn't think this matches the trained class
            if (Math.random() < 0.05) {
              console.log('Binary classification model - low confidence:', {
                modelName: this.selectedModelInfo?.name,
                trainedClass: trainedClass,
                confidence: confidence.toFixed(3),
                actualDirection: actualDirection,
                message: 'Confidence below threshold (0.5)'
              });
            }
            return null;
          }
        }
        
        // Only return prediction if:
        // 1. Model predicts the trained class (which it always will for single-class)
        // 2. For binary models: confidence is above threshold (already checked above)
        // 3. The actual finger direction matches the trained class
        // Otherwise, return null (no match)
        if (predictedClass === trainedClass && actualDirection === trainedClass) {
          // Model prediction matches actual direction - valid detection
          if (Math.random() < 0.05) {
            console.log('Single-class model validation passed:', {
              modelName: this.selectedModelInfo?.name,
              trainedClass: trainedClass,
              predicted: predictedClass,
              actualDirection: actualDirection,
              confidence: confidence.toFixed(3),
              isBinary: isBinaryClassificationModel
            });
          }
          return { direction: predictedClass, confidence: confidence };
        } else {
          // Model predicts trained class but actual direction is different - no match
          if (Math.random() < 0.05) {
            console.log('Single-class model validation failed:', {
              modelName: this.selectedModelInfo?.name,
              trainedClass: trainedClass,
              predicted: predictedClass,
              actualDirection: actualDirection,
              confidence: confidence.toFixed(3),
              isBinary: isBinaryClassificationModel,
              message: 'Finger direction does not match trained class'
            });
          }
          return null; // Return null because finger direction doesn't match trained class
        }
      }
      
      // IMPORTANT: Check if predicted class is in the trained classes
      // If model was trained on "Up", "Right", "Down", it should never predict "Left"
      if (!trainedClasses.includes(predictedClass)) {
        // Model predicted a class it wasn't trained on - invalid!
        if (!this._lastUntrainedLogTime || (Date.now() - this._lastUntrainedLogTime) > 2000) {
          // Format probabilities based on model format
          let probStr;
          if (labelMap && trainedLabels.length === probabilities.length) {
            probStr = trainedLabels.map((label, i) => `${label}:${probabilities[i].toFixed(3)}`).join(', ');
          } else {
            const oldClasses = ['up', 'down', 'left', 'right'];
            probStr = oldClasses.map((label, i) => i < probabilities.length ? `${label}:${probabilities[i].toFixed(3)}` : '').filter(s => s).join(', ');
          }
          console.warn('⚠️ Model predicted untrained class:', {
            modelName: this.selectedModelInfo?.name,
            predicted: predictedClass,
            trainedClasses: trainedClasses,
            confidence: confidence.toFixed(3),
            allProbabilities: probStr,
            message: 'Model predicted a class it was not trained on - REJECTED'
          });
          this._lastUntrainedLogTime = Date.now();
        }
        return null;
      }
      
      // CRITICAL: For multi-class models, validate that actual finger direction is in trained set
      // If finger is pointing in a direction NOT in the trained set, return null (show "--")
      // This prevents the model from always outputting one of the trained classes
      if (actualDirection && !trainedClasses.includes(actualDirection)) {
        // Actual finger direction is NOT in the trained set - reject prediction
        // The model will always predict one of its trained classes, but if the finger
        // is actually pointing in an untrained direction, we should show "--"
        if (!this._lastUntrainedDirectionLogTime || (Date.now() - this._lastUntrainedDirectionLogTime) > 2000) {
          console.log('⚠️ Multi-class model - finger pointing in untrained direction:', {
            modelName: this.selectedModelInfo?.name,
            predicted: predictedClass,
            actualDirection: actualDirection,
            trainedClasses: trainedClasses,
            confidence: confidence.toFixed(3),
            message: 'Finger direction is not in trained set - REJECTED (should show "--")'
          });
          this._lastUntrainedDirectionLogTime = Date.now();
        }
        return null;
      }
      
      // Log geometric detection for debugging
      if (!this._lastGeometricLogTime || (Date.now() - this._lastGeometricLogTime) > 2000) {
        console.log('=== Geometric Detection ===');
        console.log('Geometric detection says:', actualDirection);
        console.log('Model predicts:', predictedClass, 'with confidence:', confidence.toFixed(3));
        this._lastGeometricLogTime = Date.now();
      }
      
      // Check confidence threshold - if very low, reject immediately
      if (confidence < 0.25) {
        // Very low confidence - return null
        if (!this._lastLowConfLogTime || (Date.now() - this._lastLowConfLogTime) > 2000) {
          // Format probabilities based on model format
          let probStr;
          if (labelMap && trainedLabels.length === probabilities.length) {
            probStr = trainedLabels.map((label, i) => `${label}:${probabilities[i].toFixed(3)}`).join(', ');
          } else {
            const oldClasses = ['up', 'down', 'left', 'right'];
            probStr = oldClasses.map((label, i) => i < probabilities.length ? `${label}:${probabilities[i].toFixed(3)}` : '').filter(s => s).join(', ');
          }
          console.log('ℹ️ Very low confidence - no prediction from multi-class model:', {
            modelName: this.selectedModelInfo?.name,
            maxConfidence: confidence.toFixed(3),
            predicted: predictedClass,
            trainedClasses: trainedClasses,
            allProbabilities: probStr
          });
          this._lastLowConfLogTime = Date.now();
        }
        return null;
      }
      
      // Additional validation: if actualDirection exists and is in trained set,
      // prioritize it when model confidence is low
      if (actualDirection && trainedClasses.includes(actualDirection)) {
        // Actual direction is in trained set - validate it matches prediction
        if (actualDirection === predictedClass) {
          // Perfect match - accept prediction
          if (!this._lastPredictionLogTime || (Date.now() - this._lastPredictionLogTime) > 2000) {
            // Format probabilities based on model format
            let probStr;
            if (labelMap && trainedLabels.length === probabilities.length) {
              probStr = trainedLabels.map((label, i) => `${label}:${probabilities[i].toFixed(3)}`).join(', ');
            } else {
              const oldClasses = ['up', 'down', 'left', 'right'];
              probStr = oldClasses.map((label, i) => i < probabilities.length ? `${label}:${probabilities[i].toFixed(3)}` : '').filter(s => s).join(', ');
            }
            console.log('✓ Multi-class model prediction (validated):', {
              modelName: this.selectedModelInfo?.name,
              predicted: predictedClass,
              actualDirection: actualDirection,
              confidence: confidence.toFixed(3),
              trainedClasses: trainedClasses,
              probabilities: probStr
            });
            this._lastPredictionLogTime = Date.now();
          }
          return { direction: predictedClass, confidence: confidence };
        } else {
          // Actual direction is in trained set but doesn't match prediction
          // When confidence is low (< 0.4), trust geometric detection over model
          // When confidence is high (>= 0.4), trust model over geometric detection
          if (confidence < 0.4) {
            // Low confidence - trust geometric detection instead
            if (!this._lastLowConfMismatchLogTime || (Date.now() - this._lastLowConfMismatchLogTime) > 2000) {
              console.log('⚠️ Low confidence mismatch - trusting geometric detection:', {
                modelName: this.selectedModelInfo?.name,
                modelPredicted: predictedClass,
                geometricDetected: actualDirection,
                modelConfidence: confidence.toFixed(3),
                trainedClasses: trainedClasses,
                message: 'Low model confidence - using geometric detection instead'
              });
              this._lastLowConfMismatchLogTime = Date.now();
            }
            // Use geometric detection instead of model prediction
            return { direction: actualDirection, confidence: 0.5 }; // Use medium confidence for geometric detection
          } else {
            // High confidence - trust the model even if actualDirection differs
            // (actualDirection might be slightly off due to angle detection limitations)
            if (!this._lastHighConfMismatchLogTime || (Date.now() - this._lastHighConfMismatchLogTime) > 2000) {
              console.log('ℹ️ High confidence mismatch (trusting model):', {
                modelName: this.selectedModelInfo?.name,
                predicted: predictedClass,
                actualDirection: actualDirection,
                confidence: confidence.toFixed(3),
                trainedClasses: trainedClasses,
                message: 'High confidence - trusting model over geometric detection'
              });
              this._lastHighConfMismatchLogTime = Date.now();
            }
            return { direction: predictedClass, confidence: confidence };
          }
        }
      }
      
      // If actualDirection is null or not in trained set, but model has reasonable confidence,
      // trust the model (geometric detection might have failed)
      if (!actualDirection || !trainedClasses.includes(actualDirection)) {
        if (confidence >= 0.35) {
          // Model has reasonable confidence - trust it even if geometric detection failed
          if (!this._lastNoActualDirLogTime || (Date.now() - this._lastNoActualDirLogTime) > 2000) {
            console.log('ℹ️ Multi-class model - trusting model (no valid geometric detection):', {
              modelName: this.selectedModelInfo?.name,
              predicted: predictedClass,
              actualDirection: actualDirection,
              confidence: confidence.toFixed(3),
              trainedClasses: trainedClasses,
              message: 'Reasonable confidence - trusting model despite geometric detection issue'
            });
            this._lastNoActualDirLogTime = Date.now();
          }
          return { direction: predictedClass, confidence: confidence };
        }
      }
      
      // If actualDirection is null (hand detection failed), trust model if confidence is high
      if (!actualDirection && confidence >= 0.5) {
        // No actual direction available but high confidence - trust model
        if (!this._lastPredictionLogTime || (Date.now() - this._lastPredictionLogTime) > 2000) {
          // Format probabilities based on model format
          let probStr;
          if (labelMap && trainedLabels.length === probabilities.length) {
            probStr = trainedLabels.map((label, i) => `${label}:${probabilities[i].toFixed(3)}`).join(', ');
          } else {
            const oldClasses = ['up', 'down', 'left', 'right'];
            probStr = oldClasses.map((label, i) => i < probabilities.length ? `${label}:${probabilities[i].toFixed(3)}` : '').filter(s => s).join(', ');
          }
          console.log('ℹ️ Multi-class model prediction (no actualDirection, high confidence):', {
            modelName: this.selectedModelInfo?.name,
            predicted: predictedClass,
            actualDirection: null,
            confidence: confidence.toFixed(3),
            trainedClasses: trainedClasses,
            probabilities: probStr
          });
          this._lastPredictionLogTime = Date.now();
        }
        return { direction: predictedClass, confidence: confidence };
      }
      
      // If we get here, actualDirection is null and confidence is low - reject
      return null;
    } catch (error) {
      console.error('Error in trained model prediction:', error);
      console.error('Model ID:', this.selectedModelId);
      console.error('Model info:', this.selectedModelInfo);
      return null; // Return null instead of falling back to default detection
    }
  }

  // Test Mode Methods
  async startTestMode() {
    if (!this.trainedModel || !this.selectedModelId) {
      alert('Please select a model from the list first.');
      return;
    }

    if (!this.mediaPipeReady || !this.imageHandLandmarker) {
      alert('MediaPipe not initialized yet. Please wait...');
      return;
    }

    try {
      // Create video landmarker for test mode if not exists
      if (!this.handLandmarker) {
        this.testStatus.innerHTML = '<strong>Status:</strong> Initializing video detector...';
        const FilesetResolver = window.MediaPipeFilesetResolver;
        const HandLandmarker = window.MediaPipeHandLandmarker;
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "CPU"
          },
          numHands: 1,
          runningMode: "VIDEO",
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
      }

      // Request camera access
      this.testStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      this.testVideo.srcObject = this.testStream;
      await this.testVideo.play();
      
      // Wait for video to be ready
      await new Promise((resolve) => {
        const checkReady = () => {
          if (this.testVideo.readyState >= 2 && this.testVideo.videoWidth > 0) {
            resolve();
          } else {
            setTimeout(checkReady, 100);
          }
        };
        checkReady();
      });
      
      // Resize canvas to match video
      this.testCanvas.width = this.testVideo.videoWidth;
      this.testCanvas.height = this.testVideo.videoHeight;
      
      // Set canvas display size to match video
      const videoElement = this.testVideo;
      this.testCanvas.style.width = videoElement.style.width;
      this.testCanvas.style.height = 'auto';
      
      // Verify model is still loaded before starting
      if (!this.trainedModel || !this.selectedModelId) {
        alert('Model was lost. Please select a model again.');
        this.testStatus.innerHTML = '<strong>Status:</strong> Error: Model not available. Please select a model.';
        return;
      }
      
      // Show test camera and start detection
      this.testCameraContainer.style.display = 'block';
      this.testBtn.style.display = 'none';
      this.stopTestBtn.style.display = 'inline-block';
      this.isTestModeActive = true;
      
      // Display which model is being used
      const modelName = this.selectedModelInfo ? this.selectedModelInfo.name : 'Unknown';
      this.testStatus.innerHTML = `<strong>Status:</strong> Testing with model "<strong style="color: #4CAF50;">${modelName}</strong>". Point your finger in different directions.`;
      
      console.log('✓ Test mode started');
      console.log('✓ Using model ID:', this.selectedModelId);
      console.log('✓ Using model name:', modelName);
      console.log('✓ Model object:', this.trainedModel);
      
      // Start detection loop
      this.detectInTestMode();
    } catch (error) {
      console.error('Error starting test mode:', error);
      alert(`Error starting test camera: ${error.message}. Please check permissions.`);
      this.testStatus.innerHTML = '<strong>Status:</strong> Error starting test mode.';
    }
  }

  stopTestMode() {
    this.isTestModeActive = false;
    
    if (this.testStream) {
      const tracks = this.testStream.getTracks();
      tracks.forEach(track => track.stop());
      this.testStream = null;
    }
    
    if (this.testVideo) {
      this.testVideo.srcObject = null;
    }
    
    // Hide test camera
    this.testCameraContainer.style.display = 'none';
    this.testBtn.style.display = 'inline-block';
    this.stopTestBtn.style.display = 'none';
    
    // Clear canvas
    this.testCtx.clearRect(0, 0, this.testCanvas.width, this.testCanvas.height);
    
    // Reset display boxes to default
    this.updateTestDirectionDisplay(null, null);
    
    this.testStatus.innerHTML = `<strong>Status:</strong> Test mode stopped.`;
  }

  async detectInTestMode() {
    if (!this.isTestModeActive) return;

    const startTimeMs = performance.now();
    
    // Only process if video has advanced
    if (this.testVideo.currentTime > 0) {
      const results = this.handLandmarker.detectForVideo(this.testVideo, startTimeMs);
      this.processTestResults(results);
    }

    if (this.isTestModeActive) {
      requestAnimationFrame(() => this.detectInTestMode());
    }
  }

  processTestResults(results) {
    // Clear canvas
    this.testCtx.clearRect(0, 0, this.testCanvas.width, this.testCanvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      
      // Mirror landmarks for both drawing and detection
      // Video is mirrored (scaleX(-1)) for display, so we need to mirror landmarks for drawing
      // Model was trained on mirrored images, so we also need mirrored landmarks for detection
      const mirroredLandmarks = landmarks.map(landmark => ({
        x: 1 - landmark.x, // Mirror x coordinate (flip horizontally)
        y: landmark.y,     // Keep y coordinate the same
        z: landmark.z      // Keep z coordinate the same
      }));
      
      // Draw hand landmarks using mirrored coordinates (video is mirrored, so use mirrored landmarks)
      this.drawHandLandmarksOnCanvas(mirroredLandmarks, this.testCtx, this.testCanvas);
      
      // Detect direction using trained model with mirrored landmarks
      // (model was trained on mirrored images, so we need mirrored landmarks)
      // Pass both original and mirrored landmarks: original for validation, mirrored for model
      const result = this.detectDirectionWithTrainedModel(mirroredLandmarks, landmarks);
      const direction = result ? result.direction : null;
      const confidence = result ? result.confidence : null;
      this.updateTestDirectionDisplay(direction, confidence);
    } else {
      // No hand detected - reset to default
      this.updateTestDirectionDisplay(null, null);
    }
  }

  drawHandLandmarksOnCanvas(landmarks, ctx, canvas) {
    // Draw index finger line
    const wrist = landmarks[this.WRIST];
    const indexMCP = landmarks[this.INDEX_FINGER_MCP];
    const indexTip = landmarks[this.INDEX_FINGER_TIP];
    
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(wrist.x * canvas.width, wrist.y * canvas.height);
    ctx.lineTo(indexMCP.x * canvas.width, indexMCP.y * canvas.height);
    ctx.lineTo(indexTip.x * canvas.width, indexTip.y * canvas.height);
    ctx.stroke();
    
    // Draw index finger tip
    ctx.fillStyle = '#FF0000';
    ctx.beginPath();
    ctx.arc(indexTip.x * canvas.width, indexTip.y * canvas.height, 8, 0, 2 * Math.PI);
    ctx.fill();
  }

  updateTestDirectionDisplay(direction, confidence = null) {
    // Update direction box
    if (direction && this.testDirectionValue && this.testDirectionDisplay) {
      this.testDirectionValue.textContent = direction.toUpperCase();
      
      // Set direction box colors based on direction
      if (direction === 'up') {
        this.testDirectionDisplay.style.background = 'rgba(0, 255, 0, 0.9)';
        this.testDirectionDisplay.style.color = '#ffffff';
      } else if (direction === 'down') {
        this.testDirectionDisplay.style.background = 'rgba(255, 0, 0, 0.9)';
        this.testDirectionDisplay.style.color = '#ffffff';
      } else if (direction === 'left') {
        this.testDirectionDisplay.style.background = 'rgba(255, 165, 0, 0.9)';
        this.testDirectionDisplay.style.color = '#ffffff';
      } else if (direction === 'right') {
        this.testDirectionDisplay.style.background = 'rgba(0, 100, 255, 0.9)';
        this.testDirectionDisplay.style.color = '#ffffff';
      }
    } else {
      // No detection - default style
      if (this.testDirectionValue) {
        this.testDirectionValue.textContent = '--';
      }
      if (this.testDirectionDisplay) {
        this.testDirectionDisplay.style.background = 'rgba(0, 0, 0, 0.7)';
        this.testDirectionDisplay.style.color = '#ffffff';
      }
    }
    
    // Update confidence box
    if (confidence !== null && confidence !== undefined && this.testConfidenceValue && this.testConfidenceDisplay) {
      const confidenceDecimal = confidence.toFixed(2);
      this.testConfidenceValue.textContent = confidenceDecimal;
      
      // Set confidence box background and text color based on score value
      // Green for high (>0.7), Orange/Yellow for medium (0.5-0.7), Red for low (<0.5)
      if (confidence > 0.7) {
        // High confidence - Green background, white text
        this.testConfidenceDisplay.style.background = 'rgba(76, 175, 80, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      } else if (confidence > 0.5) {
        // Medium confidence - Orange background, white text
        this.testConfidenceDisplay.style.background = 'rgba(255, 152, 0, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      } else {
        // Low confidence - Red background, white text
        this.testConfidenceDisplay.style.background = 'rgba(244, 67, 54, 0.9)';
        this.testConfidenceValue.style.color = '#ffffff';
      }
    } else {
      // No confidence - default style
      if (this.testConfidenceValue) {
        this.testConfidenceValue.textContent = '--';
      }
      if (this.testConfidenceDisplay) {
        this.testConfidenceDisplay.style.background = 'rgba(0, 0, 0, 0.7)';
        this.testConfidenceValue.style.color = '#ffffff';
      }
    }
  }

  clearTrainingData() {
    if (confirm('Are you sure you want to clear all training data? This cannot be undone.')) {
      this.trainingData = [];
      
      // Clear current training data from localStorage (but keep saved models)
      localStorage.removeItem('handDirectionTrainingData');
      
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

  saveTrainingData() {
    try {
      // Save current training data to localStorage (for current session)
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
      
      localStorage.setItem('handDirectionTrainingData', jsonString);
      console.log('Training data saved successfully. Sample count:', this.trainingData.length);
    } catch (error) {
      console.error('Error saving training data:', error);
      if (error.name === 'QuotaExceededError') {
        alert('Error: Training data is too large to save. Please reduce the number of samples or clear old data.');
      }
    }
  }

  loadTrainingData() {
    try {
      const saved = localStorage.getItem('handDirectionTrainingData');
      if (saved) {
        const data = JSON.parse(saved);
        this.trainingData = data.trainingData || [];
        console.log(`Loaded ${this.trainingData.length} training samples from storage`);
        // Note: UI updates will happen after render() in connectedCallback
      }
    } catch (error) {
      console.error('Error loading training data:', error);
      this.trainingData = [];
    }
  }
  
  // Model Management Methods
  sanitizeModelName(name) {
    // Sanitize model name for IndexedDB (remove special characters)
    return name.replace(/[^a-zA-Z0-9_-]/g, '_');
  }
  
  async saveModel(model, modelName, trainingData, uniqueLabels = null, labelMap = null) {
    try {
      // Sanitize model name for storage
      const sanitizedName = this.sanitizeModelName(modelName);
      const storageKey = `hand_model_${sanitizedName}_${Date.now()}`;
      
      // Save model to IndexedDB
      await model.save('indexeddb://' + storageKey);
      
      // Extract only essential metadata from training data (not the full data with images)
      // This prevents localStorage quota exceeded errors
      // Use provided uniqueLabels if available, otherwise extract from trainingData
      if (!uniqueLabels) {
        uniqueLabels = [...new Set(trainingData.map(sample => sample.label))].sort();
      }
      const labelCounts = {};
      trainingData.forEach(sample => {
        labelCounts[sample.label] = (labelCounts[sample.label] || 0) + 1;
      });
      
      // Create labelMap if not provided (for backward compatibility)
      if (!labelMap) {
        labelMap = {};
        uniqueLabels.forEach((label, index) => {
          labelMap[label] = index;
        });
      }
      
      // Log what labels are being saved with this model
      console.log('=== Saving Model ===');
      console.log('Model Name:', modelName);
      console.log('Trained Labels (directions):', uniqueLabels);
      console.log('Label Mapping (index -> label):', labelMap);
      console.log('Label Counts:', labelCounts);
      console.log('Total Samples:', trainingData.length);
      
      // Save model metadata (without full training data to save space)
      const modelInfo = {
        id: Date.now().toString(),
        name: modelName, // Keep original name for display
        storageKey: storageKey, // Use sanitized key for storage
        trainingDataCount: trainingData.length,
        trainedLabels: uniqueLabels, // Store only the list of trained labels (sorted)
        labelMap: labelMap, // Store label mapping (label -> index) for prediction
        labelCounts: labelCounts, // Store label counts for reference
        createdAt: new Date().toISOString()
        // NOTE: We don't store full trainingData to avoid localStorage quota issues
        // The model itself is saved in IndexedDB, and we only need label info for validation
      };
      
      // Load existing models
      const savedModels = this.loadSavedModelsList();
      savedModels.push(modelInfo);
      
      // Check storage size before saving
      const jsonString = JSON.stringify(savedModels);
      const sizeInMB = new Blob([jsonString]).size / (1024 * 1024);
      
      if (sizeInMB > 4) {
        console.warn('Warning: Model metadata is large (', sizeInMB.toFixed(2), 'MB). Consider deleting old models.');
      }
      
      // Save to localStorage with error handling
      try {
        localStorage.setItem('handDirectionSavedModels', jsonString);
      } catch (storageError) {
        if (storageError.name === 'QuotaExceededError') {
          // Try to free up space by removing old models
          console.warn('Storage quota exceeded. Attempting to free space...');
          
          // Keep only the 10 most recent models
          const sortedModels = savedModels.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
          const recentModels = sortedModels.slice(0, 10);
          
          // Remove old models from IndexedDB
          const modelsToDelete = sortedModels.slice(10);
          for (const oldModel of modelsToDelete) {
            try {
              const oldStorageKey = oldModel.storageKey || this.sanitizeModelName(oldModel.name);
              await tf.io.removeModel('indexeddb://' + oldStorageKey);
            } catch (e) {
              console.warn('Could not remove old model from IndexedDB:', e);
            }
          }
          
          // Try saving again with only recent models
          const reducedJsonString = JSON.stringify(recentModels);
          localStorage.setItem('handDirectionSavedModels', reducedJsonString);
          
          console.log('Freed space by removing old models. Kept', recentModels.length, 'most recent models.');
        } else {
          throw storageError;
        }
      }
      
      // Reload models list
      this.loadSavedModels();
      
      return modelInfo;
    } catch (error) {
      console.error('Error saving model:', error);
      throw error;
    }
  }
  
  loadSavedModelsList() {
    try {
      const saved = localStorage.getItem('handDirectionSavedModels');
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
      
      // Load model from IndexedDB using storage key
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
      
      // Remove from list
      const updatedModels = models.filter(m => m.id !== modelId);
      localStorage.setItem('handDirectionSavedModels', JSON.stringify(updatedModels));
      
      // Try to delete from IndexedDB (may fail silently if not found)
      try {
        const storageKey = modelInfo.storageKey || this.sanitizeModelName(modelInfo.name);
        tf.io.removeModel('indexeddb://' + storageKey);
      } catch (e) {
        console.warn('Could not remove model from IndexedDB:', e);
      }
      
      // If this was the selected model, clear selection
      if (this.selectedModelId === modelId) {
        this.selectedModelId = null;
        this.trainedModel = null;
        this.selectedModelInfo = null;
        this.testBtn.disabled = true;
        // Stop test mode if active
        if (this.isTestModeActive) {
          this.stopTestMode();
        }
        this.testStatus.innerHTML = '<strong>Status:</strong> Selected model was deleted. Please select another model.';
      }
      
      // Reload models list
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
      
      // Store both the model and model info
      this.selectedModelId = modelId;
      this.trainedModel = model;
      this.selectedModelInfo = modelInfo; // Store model info for verification
      
      // Verify model is loaded
      if (!this.trainedModel) {
        throw new Error('Model loaded but is null');
      }
      
      // Enable test button
      this.testBtn.disabled = false;
      this.testStatus.innerHTML = `<strong>Status:</strong> Model "${modelInfo.name}" (${modelInfo.trainingDataCount} samples) loaded and ready. Click "Start Test" to begin testing.`;
      
      // Update models list to show selection
      this.updateModelsList();
      
      console.log('✓ Model selected and loaded:', modelInfo.name);
      console.log('✓ Model ID:', modelId);
      console.log('✓ Model training samples:', modelInfo.trainingDataCount);
      console.log('✓ Trained model object:', this.trainedModel);
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
    const counts = {
      up: 0,
      down: 0,
      left: 0,
      right: 0
    };
    
    this.trainingData.forEach(sample => {
      counts[sample.label]++;
    });
    
    const total = this.trainingData.length;
    const modelStatus = this.trainedModel ? 'Trained' : 'Not trained';
    
    this.trainingStats.innerHTML = `
      <strong>Training Data:</strong> ${total} samples 
      (UP: ${counts.up}, DOWN: ${counts.down}, LEFT: ${counts.left}, RIGHT: ${counts.right}) | 
      <strong>Model Status:</strong> ${modelStatus}
    `;
  }
  
  updateTrainingDataTable() {
    if (!this.trainingDataTable) {
      console.error('trainingDataTable element not found! Cannot update table.');
      return;
    }
    
    console.log('=== Updating training data table ===');
    console.log('Sample count:', this.trainingData ? this.trainingData.length : 0);
    console.log('Training data:', this.trainingData);
    
    if (!this.trainingData || this.trainingData.length === 0) {
      console.log('No training data, showing empty message');
      this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">No training data yet. Upload images or take screenshots to get started.</p>';
      return;
    }
    
    // Create table/list of training data
    let validItemsCount = 0;
    const itemsHTML = this.trainingData.map((sample, index) => {
      // Ensure sample has required properties
      if (!sample) {
        console.warn('Null/undefined sample at index', index);
        return '';
      }
      
      if (!sample.label) {
        console.warn('Sample missing label at index', index, sample);
        return '';
      }
      
      validItemsCount++;
      const directionColors = {
        up: '#4CAF50',
        down: '#f44336',
        left: '#FF9800',
        right: '#2196F3'
      };
      
      const directionLabels = {
        up: 'UP',
        down: 'DOWN',
        left: 'LEFT',
        right: 'RIGHT'
      };
      
      const color = directionColors[sample.label] || '#667eea';
      const label = directionLabels[sample.label] || sample.label.toUpperCase();
      
      const hasImage = sample.imageDataUrl && sample.imageDataUrl.length > 0;
      
      return `
        <div data-sample-index="${index}" style="border: 2px solid ${color}; border-radius: 8px; padding: 8px; background: white; display: flex; flex-direction: column; align-items: center; position: relative;">
          ${hasImage ? `
            <img src="${sample.imageDataUrl}" 
                 alt="Training sample ${index + 1}" 
                 style="width: 100%; height: 120px; object-fit: cover; border-radius: 4px; background: #f0f0f0;"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
          ` : ''}
          <div style="${hasImage ? 'display: none;' : 'display: flex;'} width: 100%; height: 120px; background: #f0f0f0; border-radius: 4px; align-items: center; justify-content: center; color: #999; font-size: 12px;">${hasImage ? 'No image' : 'No image available'}</div>
          <div style="margin-top: 8px; padding: 4px 8px; background: ${color}; color: white; border-radius: 4px; font-weight: bold; font-size: 12px; text-align: center; width: 100%;">
            ${label}
          </div>
          <button class="delete-sample-btn" data-index="${index}"
                  style="position: absolute; top: 4px; right: 4px; background: rgba(255,0,0,0.7); color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 14px; line-height: 1; display: flex; align-items: center; justify-content: center;"
                  title="Delete this sample">×</button>
        </div>
      `;
    }).filter(item => item !== '').join('');
    
    console.log('Valid items count:', validItemsCount);
    console.log('Generated HTML length:', itemsHTML ? itemsHTML.length : 0);
    console.log('First 200 chars of HTML:', itemsHTML ? itemsHTML.substring(0, 200) : 'empty');
    
    if (!itemsHTML || itemsHTML.length === 0) {
      console.error('No HTML generated! This should not happen if validItemsCount > 0');
      this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #f44336; text-align: center; margin: 20px 0;">Error: No valid training data to display. Check console for details.</p>';
      return;
    }
    
    try {
      this.trainingDataTable.innerHTML = itemsHTML;
      console.log('✓ Training data table updated successfully!');
    } catch (error) {
      console.error('Error setting innerHTML:', error);
      this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #f44336; text-align: center; margin: 20px 0;">Error displaying training data. Check console for details.</p>';
    }
    
    // Add event listeners for delete buttons
    this.trainingDataTable.querySelectorAll('.delete-sample-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(btn.dataset.index);
        if (confirm(`Delete this ${this.trainingData[index].label.toUpperCase()} sample?`)) {
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
customElements.define('hand-direction-detector', HandDirectionDetector);

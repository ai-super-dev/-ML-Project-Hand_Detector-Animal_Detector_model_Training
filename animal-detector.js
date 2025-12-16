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
          <div class="training-controls-row">
            <input type="text" id="animalTypeInput" placeholder="Enter animal name (e.g., cat, dog, bird)">
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
          <button id="testBtn" class="toggle-btn" disabled>Start Test</button>
          <button id="stopTestBtn" class="clear-btn" style="display: none;">Stop Test</button>
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
      // Wait for MediaPipe to be loaded
      let retries = 0;
      const maxRetries = 100;
      
      while (retries < maxRetries) {
        if (window.mediaPipeReady && window.MediaPipeFilesetResolver) {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
        retries++;
      }
      
      if (!window.MediaPipeFilesetResolver) {
        throw new Error('MediaPipe libraries not loaded');
      }

      this.statusText.textContent = 'MediaPipe initialized. Ready for training and testing.';
      this.mediaPipeReady = true;
    } catch (error) {
      console.error('Error initializing MediaPipe:', error);
      this.statusText.textContent = `Error: ${error.message}. Check console (F12) for details.`;
    }
  }

  // Placeholder methods - to be implemented later
  loadTrainingData() {
    // Load from localStorage
    const saved = localStorage.getItem('animalTrainingData');
    if (saved) {
      try {
        this.trainingData = JSON.parse(saved);
      } catch (e) {
        console.error('Error loading training data:', e);
        this.trainingData = [];
      }
    }
  }

  handleFileUpload(e) {
    // Placeholder - to be implemented
    console.log('File upload - to be implemented');
  }

  openTrainingCamera() {
    // Placeholder - to be implemented
    console.log('Open training camera - to be implemented');
  }

  takeScreenshot() {
    // Placeholder - to be implemented
    console.log('Take screenshot - to be implemented');
  }

  closeTrainingCamera() {
    // Placeholder - to be implemented
    console.log('Close training camera - to be implemented');
  }

  trainModel() {
    // Placeholder - to be implemented
    console.log('Train model - to be implemented');
  }

  clearTrainingData() {
    // Placeholder - to be implemented
    console.log('Clear training data - to be implemented');
  }

  startTestMode() {
    // Placeholder - to be implemented
    console.log('Start test mode - to be implemented');
  }

  stopTestMode() {
    // Placeholder - to be implemented
    console.log('Stop test mode - to be implemented');
  }

  loadSavedModels() {
    // Load from localStorage
    const saved = localStorage.getItem('animalSavedModels');
    if (saved) {
      try {
        this.savedModels = JSON.parse(saved);
        this.updateModelsList();
      } catch (e) {
        console.error('Error loading saved models:', e);
        this.savedModels = [];
      }
    }
  }

  updateModelsList() {
    // Placeholder - to be implemented
    if (this.modelsList) {
      if (this.savedModels.length === 0) {
        this.modelsList.innerHTML = '<p style="color: #666; margin: 0; text-align: center;">No models saved yet. Train a model to get started.</p>';
      } else {
        // Will be implemented later
        this.modelsList.innerHTML = '<p style="color: #666; margin: 0; text-align: center;">Models list - to be implemented</p>';
      }
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
    // Placeholder - to be implemented
    if (this.trainingDataTable) {
      if (this.trainingData.length === 0) {
        this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">No training data yet. Upload images or take screenshots to get started.</p>';
      } else {
        // Will be implemented later
        this.trainingDataTable.innerHTML = '<p style="grid-column: 1 / -1; color: #666; text-align: center; margin: 20px 0;">Training data display - to be implemented</p>';
      }
    }
  }
}

// Register the custom element
customElements.define('animal-detector', AnimalDetector);

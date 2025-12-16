# Hand Direction Detector Web Component

A web component that detects hand direction (up, down, left, right) using the index finger with MediaPipe Hands.

## Features

- 🖐️ Real-time hand detection using MediaPipe Hands
- 👆 Direction detection based on index finger position
- 🎨 Visual feedback with color-coded direction display
- 📱 Responsive design
- 🔄 Easy to use web component

## How It Works

The component uses MediaPipe Hands to detect hand landmarks in real-time. It specifically tracks:
- **Wrist position** - as the reference point
- **Index finger tip** - to determine the pointing direction
- **Index finger MCP joint** - to verify the finger is extended

The direction is calculated based on the vector between the wrist and index finger tip, accounting for the mirrored video display.

## Installation

No installation required! The component uses CDN links for MediaPipe libraries.

## Usage

### Basic Usage

1. Include the component script in your HTML:

```html
<script type="module" src="hand-direction-detector.js"></script>
```

2. Use the custom element:

```html
<hand-direction-detector></hand-direction-detector>
```

### Running Locally

1. Open `index.html` in a modern web browser, or
2. Use a local server:
   ```bash
   npm install
   npm start
   ```
3. Navigate to `http://localhost:8080`
4. Click "Start Detection"
5. Allow camera access when prompted
6. Point your index finger in different directions

### Using in Your Project

```html
<!DOCTYPE html>
<html>
<head>
    <title>My App</title>
</head>
<body>
    <hand-direction-detector></hand-direction-detector>
    <script type="module" src="hand-direction-detector.js"></script>
</body>
</html>
```

## Browser Requirements

- Modern browser with WebRTC support (Chrome, Firefox, Edge, Safari)
- HTTPS or localhost (required for camera access)
- ES6 modules support
- WebGL support (for MediaPipe)

## API

The component automatically handles:
- MediaPipe initialization
- Camera access
- Hand detection
- Direction calculation
- Visual feedback

### Methods (via JavaScript)

```javascript
const detector = document.querySelector('hand-direction-detector');

// Start detection programmatically
detector.startDetection();

// Stop detection programmatically
detector.stopDetection();
```

## Customization

You can customize the component by modifying the CSS in the shadow DOM or by extending the class:

```javascript
class CustomHandDetector extends HandDirectionDetector {
  detectDirection(landmarks) {
    // Your custom direction detection logic
  }
}
customElements.define('custom-hand-detector', CustomHandDetector);
```

## Troubleshooting

- **Camera not working**: Ensure you're using HTTPS or localhost
- **No hand detected**: Make sure your hand is clearly visible and well-lit
- **Wrong direction**: Try extending your index finger more clearly
- **MediaPipe loading error**: Check internet connection and browser console
- **Performance issues**: Close other applications using the camera

## License

MIT License - feel free to use in your projects!

## Credits

- Uses [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) for hand detection
- Built as a standard Web Component for easy integration

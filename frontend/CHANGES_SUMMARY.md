# Frontend Updates Summary

## ✅ Changes Implemented

### 1. Image Message Support
- **Location**: `frontend/src/App.js` (message rendering section)
- **Feature**: Messages can now display images in addition to text
- **Detection**: Messages with `type: "image"` are rendered as images
- **Implementation**: 
  - Checks for `msg.type === 'image'` in message rendering
  - Displays image using `<img>` tag with Base64 data
  - Shows optional caption below image

### 2. Chart Detection & API Routing
- **Location**: `frontend/src/App.js` (handleSubmit function)
- **Feature**: Automatically detects "chart" keyword and routes to appropriate endpoint
- **Logic**:
  ```javascript
  const isChartQuery = userMessage.toLowerCase().includes('chart');
  const apiUrl = isChartQuery ? API_CHART_URL : API_QUERY_URL;
  ```
- **Endpoints**:
  - Normal queries → `http://localhost:5000/api/query`
  - Chart queries → `http://localhost:5000/api/chart`

### 3. Base64 Image Handling
- **Location**: `frontend/src/App.js` (handleSubmit function)
- **Feature**: Handles Base64 image responses from backend
- **Supported Formats**:
  - `response.data.image` - Base64 image string
  - `response.data.chart` - Base64 chart string
- **Processing**:
  - Automatically adds `data:image/png;base64,` prefix if missing
  - Preserves existing data URI format if present

### 4. CSS Styling for Images
- **Location**: `frontend/src/App.css`
- **New Classes**:
  - `.message-image-container` - Container for image messages
  - `.message-image` - Styled image with hover effects
  - `.message-image-caption` - Caption styling below images
- **Features**:
  - Responsive image sizing (max-width: 600px)
  - Smooth hover zoom effect
  - Professional border and shadow styling
  - Matches existing financial theme

## 📍 Chat History Location

**File**: `frontend/src/App.js`

**State Declaration** (Line ~11):
```javascript
const [chatHistory, setChatHistory] = useState([]);
```

**Update Location** (Lines ~76-80):
```javascript
setChatHistory((prev) => [
  ...prev,
  [userMessage, historyEntry]
]);
```

**Format**: Array of arrays
```javascript
[
  ["user message 1", "ai response 1"],
  ["user message 2", "ai response 2"],
  ...
]
```

**Usage**: 
- Sent to backend with each API call: `chat_history: chatHistory`
- Used for maintaining conversation context
- For images, stores the caption text

## 🔧 Backend Requirements

Your backend needs to support:

1. **Normal Query Endpoint** (`/api/query`):
   ```json
   {
     "question": "user question",
     "chat_history": []
   }
   ```
   Response:
   ```json
   {
     "answer": "text response"
   }
   ```

2. **Chart Endpoint** (`/api/chart`):
   ```json
   {
     "question": "show me a chart",
     "chat_history": []
   }
   ```
   Response (either format):
   ```json
   {
     "image": "base64_string_here",
     "answer": "optional caption"
   }
   ```
   OR
   ```json
   {
     "chart": "base64_string_here",
     "caption": "optional caption"
   }
   ```

## 🎨 Example Usage

### Text Query:
```
User: "What is Apple's revenue?"
→ Routes to: /api/query
→ Displays: Text response in chat bubble
```

### Chart Query:
```
User: "Show me a chart of revenue trends"
→ Routes to: /api/chart
→ Displays: Image in chat bubble with caption
```

## 📝 Example Questions Updated

Added chart-related examples:
- "Show me a chart of capital expenditure evolution"
- "Create a chart showing revenue trends"

## 🚀 Testing

To test the new features:

1. **Text Messages**: Ask any normal question
   - Should display as text in chat bubble

2. **Chart Messages**: Include "chart" in your question
   - Should route to `/api/chart` endpoint
   - Should display image if backend returns Base64 image

3. **Chat History**: 
   - Check browser console/network tab to see `chat_history` being sent
   - History accumulates as you chat
   - Clears when you click "Clear" button

## ⚠️ Important Notes

- Chat history is **in-memory only** (resets on page refresh)
- Image messages store caption in chat history, not the image data
- Backend must return Base64 images for chart queries
- Case-insensitive "chart" detection (works for "Chart", "CHART", etc.)


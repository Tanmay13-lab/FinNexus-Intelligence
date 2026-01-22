# Chat History Location & Management

## 📍 Where is Chat History?

Chat history is managed in the **`App.js`** component using React state:

### State Declaration
```javascript
const [chatHistory, setChatHistory] = useState([]);
```
**Location**: Line 11 in `frontend/src/App.js`

### How It Works

1. **Initialization**: Chat history starts as an empty array `[]`

2. **Storage Format**: 
   ```javascript
   [
     ["user message 1", "ai response 1"],
     ["user message 2", "ai response 2"],
     ...
   ]
   ```

3. **Update Location**: Chat history is updated in the `handleSubmit` function (around lines 46-49):
   ```javascript
   setChatHistory((prev) => [
     ...prev,
     [userMessage, historyEntry]
   ]);
   ```

4. **Usage**: 
   - Sent to backend API with each request: `chat_history: chatHistory`
   - Used for context-aware responses
   - For image messages, stores the caption/text representation

### Important Notes

- **Chat history is stored in memory only** - it resets when you refresh the page
- **Cleared when**: User clicks the "Clear" button (calls `setChatHistory([])`)
- **For images**: Stores the caption or text description, not the image data itself

### To Persist Chat History

If you want to save chat history permanently, you could:
1. Use `localStorage` to save/load on page load
2. Send to backend for server-side storage
3. Use a database or session storage

### Current Implementation

- ✅ Chat history is maintained in component state
- ✅ Sent to backend with each API call
- ✅ Updated after each successful response
- ❌ Not persisted (lost on page refresh)
- ❌ Not displayed in UI (only used for context)


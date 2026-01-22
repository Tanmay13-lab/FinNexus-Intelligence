# Financial Analyzer Frontend

Modern, responsive React frontend for the Hybrid Graph RAG Financial Analyzer.

## Features

- 🎨 **Modern UI Design** - Professional financial theme with gradient accents
- 💬 **Chat Interface** - Clean, intuitive chat UI for querying financial data
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- ⚡ **Real-time Updates** - Live typing indicators and smooth animations
- 🎯 **Example Questions** - Quick-start suggestions for common queries
- 🌈 **Beautiful Animations** - Smooth transitions and visual effects

## Setup

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm start
   ```

   The app will open at [http://localhost:3000](http://localhost:3000)

3. **Make sure the backend is running**
   - The frontend expects the backend API at `http://localhost:5000`
   - Start the backend API before using the frontend:
     ```bash
     cd ../backend_api
     python backend_api.py
     ```

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Configuration

The API URL is configured in `src/App.js`. To change it, modify:

```javascript
const API_URL = 'http://localhost:5000/api/query';
```

## Design Theme

- **Primary Colors**: Blue gradient (#4a90e2 to #357abd)
- **Background**: Dark gradient (#0f1419 to #1a1f2e)
- **Typography**: Inter (sans-serif) and JetBrains Mono (monospace)
- **Style**: Modern, professional, financial-focused

## Technologies

- React 18
- Axios for API calls
- CSS3 with modern animations
- Responsive design principles


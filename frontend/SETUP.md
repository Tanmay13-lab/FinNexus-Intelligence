# Quick Setup Guide for VS Code

## ⚠️ Important: You CANNOT just copy-paste and run!

You need to follow these steps:

## Prerequisites

1. **Node.js installed** (version 14 or higher)
   - Check if installed: Open terminal and run `node --version`
   - If not installed: Download from [nodejs.org](https://nodejs.org/)

2. **Backend API running**
   - Make sure your FastAPI backend is running at `http://localhost:5000`

## Step-by-Step Setup

### 1. Open VS Code
- Open the `frontend` folder in VS Code
- Or open the entire project and navigate to the `frontend` folder

### 2. Open Terminal in VS Code
- Press `Ctrl + `` (backtick) or go to `Terminal > New Terminal`
- Make sure you're in the `frontend` directory

### 3. Install Dependencies (REQUIRED!)
```bash
npm install
```
This will create a `node_modules` folder with all required packages.
**This step is MANDATORY - the app won't work without it!**

### 4. Start the Development Server
```bash
npm start
```

The app will automatically open in your browser at `http://localhost:3000`

## File Structure (Make sure you have all these files)

```
frontend/
├── package.json          ✅ Required
├── .gitignore           ✅ Required
├── public/
│   └── index.html       ✅ Required
└── src/
    ├── index.js         ✅ Required
    ├── index.css        ✅ Required
    ├── App.js           ✅ Required
    └── App.css          ✅ Required
```

## Common Issues & Solutions

### Issue 1: "npm: command not found"
**Solution**: Node.js is not installed. Install it from nodejs.org

### Issue 2: "Cannot find module 'react'"
**Solution**: You forgot to run `npm install`. Run it now!

### Issue 3: "Port 3000 is already in use"
**Solution**: 
- Close other React apps running on port 3000
- Or change the port: Create `.env` file in `frontend/` with `PORT=3001`

### Issue 4: "Network Error" or "Failed to fetch"
**Solution**: 
- Make sure backend is running: `cd ../backend_api && python backend_api.py`
- Check if backend is at `http://localhost:5000`

### Issue 5: "Module not found" errors
**Solution**: 
- Delete `node_modules` folder
- Delete `package-lock.json` (if exists)
- Run `npm install` again

## Quick Commands Reference

```bash
# Install dependencies (do this first!)
npm install

# Start development server
npm start

# Build for production
npm run build

# Check Node.js version
node --version

# Check npm version
npm --version
```

## What Happens When You Run `npm install`?

- Downloads React, React-DOM, and other dependencies
- Creates `node_modules/` folder (this is large, ~200MB)
- Creates `package-lock.json` file
- Sets up the development environment

## After Setup

Once `npm start` is running:
- ✅ Browser opens automatically
- ✅ Hot reload is enabled (changes auto-refresh)
- ✅ Console shows any errors
- ✅ App connects to backend at `http://localhost:5000`

## Need Help?

If you encounter any errors:
1. Check the terminal output for error messages
2. Make sure Node.js is installed: `node --version`
3. Make sure you ran `npm install` first
4. Check that backend is running
5. Check browser console (F12) for errors



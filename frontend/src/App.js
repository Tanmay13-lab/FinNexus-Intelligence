import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const API_QUERY_URL = 'http://localhost:5000/api/query';
const API_CHART_URL = 'http://localhost:5000/api/chart';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { type: 'user', content: userMessage }]);
    setLoading(true);

    try {
      // Detect if message contains "chart" keyword (case-insensitive)
      const isChartQuery = userMessage.toLowerCase().includes('chart');
      const apiUrl = isChartQuery ? API_CHART_URL : API_QUERY_URL;

      const response = await axios.post(apiUrl, {
        question: userMessage,
        chat_history: chatHistory,
      });

      // Check if response contains an image (Base64)
      const responseData = response.data;
      let messageType = 'ai';
      let messageContent = '';
      let imageData = null;
      let caption = null;

      // Check for errors first
      if (responseData.error && !responseData.image) {
        // Chart generation failed or data not available
        messageContent = responseData.answer || responseData.error || 'Unable to generate chart. The requested data may not be available.';
        messageType = 'ai';
      } else if (responseData.image || responseData.chart) {
        // Handle Base64 image response
        const base64Image = responseData.image || responseData.chart;
        imageData = base64Image.startsWith('data:image') 
          ? base64Image 
          : `data:image/png;base64,${base64Image}`;
        
        caption = responseData.answer || responseData.caption || 'Chart visualization';
        messageType = 'image';
        messageContent = caption;
      } else {
        // Handle normal text response
        messageContent = responseData.answer || 'No response received.';
      }

      setMessages((prev) => [...prev, { 
        type: messageType, 
        content: messageContent,
        ...(imageData && { imageData: imageData }),
        ...(caption && { caption: caption })
      }]);
      
      // Update chat history for context
      // For images, store the caption or a text representation
      const historyEntry = messageType === 'image' 
        ? caption 
        : messageContent;
      
      setChatHistory((prev) => [
        ...prev,
        [userMessage, historyEntry]
      ]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.answer || 
                          error.response?.data?.error ||
                          error.message || 
                          'Failed to get response. Please check if the backend is running.';
      setMessages((prev) => [...prev, { 
        type: 'ai', 
        content: errorMessage,
        isError: true 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setChatHistory([]);
    inputRef.current?.focus();
  };

  const exampleQuestions = [
    "What is Apple's revenue for 2024?",
    "What are the main risks mentioned in the annual report?",
    "Show me a chart of capital expenditure evolution",
    "What is the geographical revenue distribution?",
    "Create a chart showing revenue trends",
  ];

  const handleExampleClick = (question) => {
    setInput(question);
    inputRef.current?.focus();
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div className="logo-text">
              <h1>Financial Analyzer</h1>
              <p>Hybrid Graph RAG System</p>
            </div>
          </div>
          <div className="header-stats">
            <div className="stat-item">
              <span className="stat-label">Neo4j</span>
              <span className="stat-value">Graph DB</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">LangChain</span>
              <span className="stat-value">RAG</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Gemini</span>
              <span className="stat-value">AI</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="chat-container">
          {/* Messages Area */}
          <div className="messages-area">
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <div className="welcome-icon">
                  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M12 15V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <h2>Welcome to Financial Analyzer</h2>
                <p>Ask questions about Apple's 2024 Annual Report</p>
                <p className="welcome-subtitle">Powered by Hybrid Graph RAG with Neo4j & LangChain</p>
                
                <div className="example-questions">
                  <p className="examples-label">Try asking:</p>
                  <div className="example-buttons">
                    {exampleQuestions.map((q, idx) => (
                      <button
                        key={idx}
                        className="example-btn"
                        onClick={() => handleExampleClick(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="messages-list">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`message ${msg.type} ${msg.isError ? 'error' : ''}`}
                  >
                    <div className="message-avatar">
                      {msg.type === 'user' ? (
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      ) : (
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </div>
                    <div className="message-content">
                      <div className="message-header">
                        <span className="message-author">
                          {msg.type === 'user' ? 'You' : 'Financial Analyzer'}
                        </span>
                      </div>
                      {msg.type === 'image' ? (
                        <div className="message-image-container">
                          <img 
                            src={msg.imageData || msg.content?.imageData} 
                            alt={msg.caption || msg.content?.caption || 'Chart'} 
                            className="message-image"
                          />
                          {msg.caption && (
                            <div className="message-image-caption">
                              {msg.caption}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="message-text">
                          {typeof msg.content === 'string' 
                            ? msg.content.split('\n').map((line, i) => (
                                <React.Fragment key={i}>
                                  {line}
                                  {i < msg.content.split('\n').length - 1 && <br />}
                                </React.Fragment>
                              ))
                            : msg.content
                          }
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="message ai loading">
                    <div className="message-avatar">
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="input-area">
            <form onSubmit={handleSubmit} className="input-form">
              <div className="input-wrapper">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about Apple's financial data, risks, revenue, or any aspect of the annual report..."
                  className="chat-input"
                  disabled={loading}
                />
                <div className="input-actions">
                  {messages.length > 0 && (
                    <button
                      type="button"
                      onClick={handleClear}
                      className="clear-btn"
                      title="Clear conversation"
                    >
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                  )}
                  <button
                    type="submit"
                    className="send-btn"
                    disabled={!input.trim() || loading}
                  >
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      </main>

      {/* Background Effects */}
      <div className="bg-effects">
        <div className="bg-circle circle-1"></div>
        <div className="bg-circle circle-2"></div>
        <div className="bg-circle circle-3"></div>
      </div>
    </div>
  );
}

export default App;


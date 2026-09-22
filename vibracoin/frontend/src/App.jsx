import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header.jsx';
import CoinResult from './components/CoinResult.jsx';
import ConfidenceBar from './components/ConfidenceBar.jsx';
import EventStatus from './components/EventStatus.jsx';
import VibrationGraph from './components/VibrationGraph.jsx';
import RecentDetections from './components/RecentDetections.jsx';

export default function App() {
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // 'online' | 'connecting' | 'offline'
  const [prediction, setPrediction] = useState(null);
  const [isImpact, setIsImpact] = useState(false);
  const [recentDetections, setRecentDetections] = useState([]);
  const [vibrationData, setVibrationData] = useState([]);

  const wsRef = useRef(null);
  const impactTimeoutRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // Trigger Malayalam Text-To-Speech (TTS)
  const speakMalayalam = useCallback((text) => {
    if (!('speechSynthesis' in window) || !text) return;
    try {
      window.speechSynthesis.cancel(); // Stop any ongoing speech
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      // Select Malayalam voice if available on system, otherwise fallback to default voice
      const voices = window.speechSynthesis.getVoices();
      const mlVoice = voices.find(v => v.lang.includes('ml') || v.lang.includes('hi') || v.name.toLowerCase().includes('malayalam'));
      if (mlVoice) {
        utterance.voice = mlVoice;
      } else {
        utterance.lang = 'ml-IN';
      }

      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn('[TTS ERROR]', err);
    }
  }, []);

  // Handle incoming prediction event
  const handlePrediction = useCallback((data) => {
    const enrichedData = {
      ...data,
      id: data.id || `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`
    };

    setPrediction(enrichedData);
    setIsImpact(true);

    // Speak Malayalam quirky reaction
    if (enrichedData.malayalamText) {
      speakMalayalam(enrichedData.malayalamText);
    }

    // Add to recent detections list (newest first, max 5, deduplicating by id)
    setRecentDetections((prev) => {
      if (prev.some((item) => item.id === enrichedData.id)) {
        return prev;
      }
      return [enrichedData, ...prev].slice(0, 5);
    });

    // Clear previous impact animation timeout
    if (impactTimeoutRef.current) {
      clearTimeout(impactTimeoutRef.current);
    }

    // Reset impact animation state after 1.8s
    impactTimeoutRef.current = setTimeout(() => {
      setIsImpact(false);
    }, 1800);
  }, [speakMalayalam]);

  // Initialize WebSocket connection with auto-reconnect
  const connectWebSocket = useCallback(() => {
    setConnectionStatus('connecting');

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.hostname}:5000`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[VIBRACOIN] Connected to WebSocket server');
        setConnectionStatus('online');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'prediction') {
            handlePrediction(data);
          }
        } catch (err) {
          console.error('[VIBRACOIN] Error parsing WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        console.warn('[VIBRACOIN] WebSocket disconnected. Retrying in 3s...');
        setConnectionStatus('offline');
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.error('[VIBRACOIN] WebSocket error encountered');
        ws.close();
      };
    } catch (err) {
      console.error('[VIBRACOIN] Failed to create WebSocket:', err);
      setConnectionStatus('offline');
      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, 3000);
    }
  }, [handlePrediction]);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (impactTimeoutRef.current) clearTimeout(impactTimeoutRef.current);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [connectWebSocket]);

  // Simulation API trigger
  const triggerSimulation = async (coinLabel = null) => {
    try {
      const body = coinLabel ? JSON.stringify({ coin: coinLabel }) : JSON.stringify({});
      const res = await fetch('http://localhost:5000/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }
      
      const data = await res.json();
      console.log('[VIBRACOIN] Simulation response:', data);
    } catch (err) {
      console.error('[VIBRACOIN] Simulation request failed:', err);
      // Fallback local simulation if backend API is offline
      const mockCoins = ['1rup', '2rup', '5rup', '10rup', '20rup'];
      const chosen = coinLabel || mockCoins[Math.floor(Math.random() * mockCoins.length)];
      const map = { '1rup': '₹1', '2rup': '₹2', '5rup': '₹5', '10rup': '₹10', '20rup': '₹20' };
      const fallbackPrediction = {
        type: 'prediction',
        coin: map[chosen],
        label: chosen,
        confidence: parseFloat((Math.random() * (0.98 - 0.75) + 0.75).toFixed(2)),
        message: 'Offline simulation mode active.',
        impactStrength: Math.floor(Math.random() * 40) + 60,
        secondaryImpact: Math.random() > 0.4,
        secondaryImpactDelay: 120,
        timestamp: new Date().toISOString(),
        isSimulated: true
      };
      handlePrediction(fallbackPrediction);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <Header connectionStatus={connectionStatus} />

      {/* Main Dashboard Grid */}
      <div className="dashboard-grid">
        {/* Left Column: Primary Visualization & Results */}
        <div className="left-panel">
          <CoinResult prediction={prediction} isImpact={isImpact} />
          <VibrationGraph data={vibrationData} isImpact={isImpact} />
        </div>

        {/* Right Column: Analytics & Dev Controls */}
        <div className="right-panel">
          <ConfidenceBar confidence={prediction ? prediction.confidence : 0} />
          <EventStatus prediction={prediction} />
          <RecentDetections detections={recentDetections} />

          {/* Simulation & Test Controls */}
          <div className="glass-card sim-controls-card">
            <div className="sim-header">
              <div className="section-title" style={{ marginBottom: 0 }}>
                Hardware Simulator
              </div>
              <span className="sim-badge">TEST MODE</span>
            </div>

            <button 
              className="sim-btn-main"
              onClick={() => triggerSimulation()}
            >
              <span>⚡ SIMULATE COIN DROP</span>
            </button>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', marginBottom: '0.5rem', fontFamily: 'var(--font-mono)' }}>
              TEST INDIVIDUAL DENOMINATIONS:
            </div>

            <div className="sim-coin-buttons">
              <button className="sim-coin-btn" onClick={() => triggerSimulation('1rup')}>₹1</button>
              <button className="sim-coin-btn" onClick={() => triggerSimulation('2rup')}>₹2</button>
              <button className="sim-coin-btn" onClick={() => triggerSimulation('5rup')}>₹5</button>
              <button className="sim-coin-btn" onClick={() => triggerSimulation('10rup')}>₹10</button>
              <button className="sim-coin-btn" onClick={() => triggerSimulation('20rup')}>₹20</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React from 'react';

const COIN_NAMES = {
  '₹1': '1 Rupee Coin',
  '₹2': '2 Rupee Coin',
  '₹5': '5 Rupee Coin',
  '₹10': '10 Rupee Coin',
  '₹20': '20 Rupee Coin'
};

export default function CoinResult({ prediction, isImpact }) {
  const hasPrediction = Boolean(prediction && prediction.coin);

  return (
    <div className="glass-card plate-stage-card">
      {isImpact && (
        <div className="impact-alert-banner">
          IMPACT DETECTED
        </div>
      )}

      <div className="plate-container">
        {/* Animated concentric ripple waves */}
        <div className={`sensor-plate ${isImpact ? 'impacting' : 'idle'}`}>
          <div className="ripple-wave wave-1"></div>
          <div className="ripple-wave wave-2"></div>
          <div className="ripple-wave wave-3"></div>

          {hasPrediction ? (
            <div className={`coin-display-symbol ${isImpact ? 'impact-flash' : ''}`}>
              {prediction.coin}
            </div>
          ) : (
            <div className="plate-idle-text">
              WAITING...
            </div>
          )}
        </div>
      </div>

      {hasPrediction ? (
        <div className="coin-details-group">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '4px' }}>
            <div className="coin-denomination-name">
              {COIN_NAMES[prediction.coin] || `${prediction.coin} Coin`}
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap', gap: '8px', fontSize: '0.85rem', color: 'var(--primary-cyan)', fontFamily: 'var(--font-mono)' }}>
            <span>{Math.round(prediction.confidence * 100)}% confidence</span>
            <span style={{ 
              padding: '2px 8px', 
              borderRadius: '4px', 
              fontSize: '0.7rem', 
              fontWeight: '700',
              letterSpacing: '0.05em',
              background: prediction.isSimulated ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.2)',
              color: prediction.isSimulated ? '#f59e0b' : '#10b981',
              border: `1px solid ${prediction.isSimulated ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.4)'}`
            }}>
              {prediction.source || (prediction.isSimulated ? 'SIMULATED' : 'HARDWARE SENSOR')}
            </span>

            {prediction.aiSource && (
              <span style={{ 
                padding: '2px 8px', 
                borderRadius: '4px', 
                fontSize: '0.7rem', 
                fontWeight: '700',
                background: 'rgba(99, 102, 241, 0.15)',
                color: '#818cf8',
                border: '1px solid rgba(99, 102, 241, 0.3)'
              }}>
                🤖 {prediction.aiSource}
              </span>
            )}
          </div>

          {/* Malayalam Quirky Quote Box with TTS Voice Replay */}
          {prediction.malayalamText ? (
            <div className="quirky-quote-box" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(129, 140, 248, 0.3)' }}>
              <div style={{ fontSize: '1.15rem', color: '#f8fafc', fontWeight: '600', marginBottom: '6px', lineHeight: '1.5' }}>
                "{prediction.malayalamText}"
              </div>
              {prediction.englishText && (
                <div style={{ fontSize: '0.85rem', color: '#94a3b8', fontStyle: 'italic' }}>
                  Translation: "{prediction.englishText}"
                </div>
              )}
              <button
                onClick={() => {
                  if ('speechSynthesis' in window && prediction.malayalamText) {
                    window.speechSynthesis.cancel();
                    const u = new SpeechSynthesisUtterance(prediction.malayalamText);
                    u.lang = 'ml-IN';
                    window.speechSynthesis.speak(u);
                  }
                }}
                style={{
                  marginTop: '8px',
                  background: 'rgba(129, 140, 248, 0.2)',
                  border: '1px solid rgba(129, 140, 248, 0.4)',
                  color: '#a5b4fc',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)'
                }}
              >
                🔊 REPLAY MALAYALAM AUDIO
              </button>
            </div>
          ) : prediction.message ? (
            <div className="quirky-quote-box">
              "{prediction.message}"
            </div>
          ) : null}
        </div>
      ) : (
        <div className="plate-idle-text" style={{ fontSize: '0.95rem', letterSpacing: '0.15em' }}>
          DROP A COIN ON THE PLATE
        </div>
      )}
    </div>
  );
}

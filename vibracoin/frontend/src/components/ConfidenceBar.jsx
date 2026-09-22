import React from 'react';

export default function ConfidenceBar({ confidence }) {
  const percentage = Math.round((confidence || 0) * 100);

  return (
    <div className="glass-card confidence-container">
      <div className="section-title" style={{ marginBottom: 0 }}>
        Model Confidence
      </div>

      <div className="confidence-header">
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Classification Score
        </span>
        <span className="confidence-percentage">
          {percentage}%
        </span>
      </div>

      <div className="confidence-bar-wrapper">
        <div 
          className="confidence-bar-fill"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

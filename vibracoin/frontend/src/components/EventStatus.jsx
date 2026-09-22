import React from 'react';

export default function EventStatus({ prediction }) {
  const impactStrength = prediction ? prediction.impactStrength : 0;
  const secondaryImpact = prediction ? prediction.secondaryImpact : false;
  const secondaryImpactDelay = prediction ? prediction.secondaryImpactDelay : 0;

  return (
    <div className="glass-card">
      <div className="section-title">
        Vibration Metrics
      </div>

      <div className="impact-info-grid">
        <div className="impact-info-item">
          <span className="impact-info-label">Impact Strength</span>
          <span className="impact-info-value highlight">
            {prediction ? `${impactStrength}%` : '--'}
          </span>
        </div>

        <div className="impact-info-item">
          <span className="impact-info-label">Secondary Impact</span>
          <span className="impact-info-value">
            {prediction ? (secondaryImpact ? 'Detected' : 'Not detected') : '--'}
          </span>
        </div>

        <div className="impact-info-item">
          <span className="impact-info-label">Secondary Delay</span>
          <span className="impact-info-value">
            {prediction && secondaryImpact ? `${secondaryImpactDelay} ms` : '--'}
          </span>
        </div>
      </div>
    </div>
  );
}

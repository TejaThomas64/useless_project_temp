import React from 'react';

function formatTimestamp(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  } catch (e) {
    return isoString;
  }
}

export default function RecentDetections({ detections }) {
  const list = Array.isArray(detections) ? detections.slice(0, 5) : [];

  return (
    <div className="glass-card">
      <div className="section-title">
        Recent Detections
      </div>

      {list.length === 0 ? (
        <div style={{ color: 'var(--text-subtle)', fontSize: '0.85rem', fontStyle: 'italic', padding: '0.5rem 0' }}>
          No detections logged yet. Drop a coin or trigger simulation.
        </div>
      ) : (
        <ul className="recent-list">
          {list.map((item, index) => (
            <li key={item.id || `${item.timestamp || 'item'}-${index}`} className="recent-item">
              <span className="recent-coin-badge">{item.coin}</span>
              <span className="recent-confidence">
                {Math.round(item.confidence * 100)}%
              </span>
              <span className="recent-time">
                {formatTimestamp(item.timestamp)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

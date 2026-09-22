import React from 'react';

export default function Header({ connectionStatus }) {
  // connectionStatus: 'online' | 'connecting' | 'offline'

  const getStatusBadge = () => {
    switch (connectionStatus) {
      case 'online':
        return (
          <div className="connection-badge online">
            <span className="status-dot online"></span>
            SYSTEM ONLINE
          </div>
        );
      case 'connecting':
        return (
          <div className="connection-badge connecting">
            <span className="status-dot connecting"></span>
            CONNECTING...
          </div>
        );
      case 'offline':
      default:
        return (
          <div className="connection-badge offline">
            <span className="status-dot offline"></span>
            WAITING FOR BACKEND
          </div>
        );
    }
  };

  return (
    <header className="header-container">
      <div className="header-title-group">
        <h1>
          VIBRACOIN
        </h1>
        <p className="header-subtitle">Identifying coins through vibration</p>
      </div>
      {getStatusBadge()}
    </header>
  );
}

import React, { useRef, useEffect } from 'react';

export default function VibrationGraph({ data, isImpact }) {
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const phaseRef = useRef(0);
  const amplitudeRef = useRef(0.1);

  // Trigger burst of amplitude when impact occurs
  useEffect(() => {
    if (isImpact) {
      amplitudeRef.current = 1.0;
    }
  }, [isImpact]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const render = () => {
      // Resize canvas to match display size
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);

      // Background grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < width; x += 30) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y < height; y += 20) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Center baseline
      const centerY = height / 2;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Plot waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#00f2fe';
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#00f2fe';

      ctx.beginPath();

      if (Array.isArray(data) && data.length > 0) {
        // Render passed array data (XYZ or raw samples)
        const step = width / (data.length - 1 || 1);
        data.forEach((sample, i) => {
          const val = typeof sample === 'number' ? sample : (sample.z || sample.y || sample.x || 0);
          // Scale sample to height range
          const y = centerY - (val * (height * 0.4));
          if (i === 0) ctx.moveTo(0, y);
          else ctx.lineTo(i * step, y);
        });
      } else {
        // Generate dynamic synthetic waveform (decaying sine wave + micro-noise)
        phaseRef.current += 0.15;
        
        // Slowly decay amplitude back to noise floor ~0.08
        if (amplitudeRef.current > 0.08) {
          amplitudeRef.current *= 0.96;
        }

        const points = 150;
        const step = width / points;

        for (let i = 0; i <= points; i++) {
          const x = i * step;
          const normalizedX = i / points;
          
          // Decaying envelope across graph width during impact
          const envelope = Math.sin(normalizedX * Math.PI);
          const noise = (Math.random() - 0.5) * 0.05;
          const wave1 = Math.sin(normalizedX * 25 + phaseRef.current) * 0.6;
          const wave2 = Math.sin(normalizedX * 50 - phaseRef.current * 1.5) * 0.4;
          
          const val = (wave1 + wave2 + noise) * amplitudeRef.current * envelope;
          const y = centerY - (val * (height * 0.42));

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
      ctx.shadowBlur = 0; // Reset glow

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [data]);

  return (
    <div className="glass-card">
      <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Vibration Oscilloscope</span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)' }}>
          {isImpact ? 'BURST DETECTED' : 'LIVE MONITOR'}
        </span>
      </div>

      <div className="graph-container">
        <canvas ref={canvasRef} className="vibration-canvas" />
      </div>
    </div>
  );
}

const http = require('http');
const express = require('express');
const cors = require('cors');
const { WebSocketServer, WebSocket } = require('ws');
const net = require('net');
require('dotenv').config();

const { generateMalayalamResponse } = require('./ai_generator');

const app = express();
const PORT = process.env.PORT || 5000;
const TCP_PORT = process.env.TCP_PORT || 5001;

// Middleware
app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.text({ type: ['text/plain', 'text/csv'] }));
app.use(express.urlencoded({ extended: true }));

// Create HTTP server & attach WebSocket server
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Allowed coins mapping
const COIN_MAP = {
  '1rup': '₹1',
  '2rup': '₹2',
  '5rup': '₹5',
  '10rup': '₹10',
  '20rup': '₹20'
};

// Quirky coin responses
const COIN_RESPONSES = {
  '1rup': [
    'Starting small, I see.',
    '₹1 has entered the arena.',
    'Small coin, big vibration.',
    'One rupee. Respectfully detected.'
  ],
  '2rup': [
    'Two rupees has arrived.',
    '₹2 made its entrance.',
    'Two rupees. Nice and simple.',
    'The plate heard that.'
  ],
  '5rup': [
    'The heavyweight has entered the plate.',
    '₹5 did not arrive quietly.',
    'That one definitely made an entrance.',
    'The plate felt that one.'
  ],
  '10rup': [
    'Okay, that had some energy.',
    '₹10 has landed.',
    'Ten rupees detected.',
    'That was a confident entrance.'
  ],
  '20rup': [
    '₹20 just made an entrance.',
    'Things are getting serious.',
    'Twenty rupees detected.',
    'The plate knows its worth.'
  ]
};

// Helper: Pick random element from array
function getRandomItem(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Helper: Broadcast message to all connected WebSocket clients
function broadcast(data) {
  const payload = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  });
}

// Normalizer for coin denomination input (handles '1rup', '5', 5, '₹5', etc.)
function normalizeCoinLabel(input) {
  if (!input) return null;
  const str = String(input).trim().toLowerCase().replace('₹', '');
  if (COIN_MAP[str]) return str;
  const numMap = { '1': '1rup', '2': '2rup', '5': '5rup', '10': '10rup', '20': '20rup' };
  if (numMap[str]) return numMap[str];
  return null;
}

// Helper: Process and normalize prediction data with AI Malayalam quotes
async function processPrediction({ coin, confidence, sensorValue = null, source = 'Hardware Sensor', isSimulated = false }) {
  const normalizedCoin = normalizeCoinLabel(coin) || '5rup';
  const coinSymbol = COIN_MAP[normalizedCoin];

  const validConfidence = typeof confidence === 'number' && confidence >= 0 && confidence <= 1
    ? parseFloat(confidence.toFixed(2))
    : (typeof confidence === 'string' && !isNaN(parseFloat(confidence)))
      ? parseFloat(parseFloat(confidence).toFixed(2))
      : 0.90;

  // Generate Malayalam AI response (Ollama Liquid LFM / Gemini / Fallback)
  let malayalamRes;
  try {
    malayalamRes = await generateMalayalamResponse(normalizedCoin, validConfidence);
  } catch (err) {
    malayalamRes = {
      malayalamText: "ദാ വന്നു ഒരു നാണയം! പ്ലേറ്റ് ഹരത്തിലായി!",
      englishText: "A coin has arrived! The plate is excited!",
      source: "VibraCoin Core"
    };
  }

  // Impact metrics derived from actual sensorValue or realistically simulated
  const rawNum = typeof sensorValue === 'number' ? sensorValue : parseInt(sensorValue, 10);
  const impactStrength = !isNaN(rawNum)
    ? Math.min(100, Math.max(30, Math.round((rawNum / 1024) * 100)))
    : Math.floor(Math.random() * (98 - 60 + 1)) + 60;

  const secondaryImpact = Math.random() > 0.35;
  const secondaryImpactDelay = secondaryImpact ? Math.floor(Math.random() * (220 - 90 + 1)) + 90 : 0;
  const timestamp = new Date().toISOString();

  const predictionEvent = {
    id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    type: 'prediction',
    coin: coinSymbol,
    label: normalizedCoin,
    confidence: validConfidence,
    message: malayalamRes.englishText,
    malayalamText: malayalamRes.malayalamText,
    englishText: malayalamRes.englishText,
    aiSource: malayalamRes.source,
    impactStrength,
    secondaryImpact,
    secondaryImpactDelay,
    sensorValue: !isNaN(rawNum) ? rawNum : null,
    source: isSimulated ? 'Simulated' : source,
    timestamp,
    isSimulated
  };

  // Broadcast live prediction via WebSocket to UI
  broadcast(predictionEvent);

  return predictionEvent;
}

// WebSocket Connection Handler
wss.on('connection', (ws) => {
  console.log('[WEBSOCKET] Client connected');
  ws.send(JSON.stringify({ type: 'connected', message: 'Connected to VIBRACOIN WebSocket Server' }));

  ws.on('close', () => {
    console.log('[WEBSOCKET] Client disconnected');
  });

  ws.on('error', (err) => {
    console.error('[WEBSOCKET ERROR]', err);
  });
});

// REST Endpoints

// GET /api/health
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', hardwareReady: true, aiReady: true });
});

// POST /api/predict & POST /api/sensor - Hardware friendly ingestion endpoint
const handleHardwareIngest = async (req, res) => {
  try {
    let coin, confidence, sensorValue, source;

    if (typeof req.body === 'object' && req.body !== null) {
      coin = req.body.coin || req.body.label || req.body.prediction;
      confidence = req.body.confidence || req.body.score;
      sensorValue = req.body.sensorValue || req.body.rawValue || req.body.val || req.body.adc;
      source = req.body.source || 'ESP32 MPU6050';
    } else if (typeof req.body === 'string') {
      // Plain text CSV or raw string like "5rup,0.92" or "5rup"
      const parts = req.body.trim().split(',');
      coin = parts[0];
      confidence = parts[1] ? parseFloat(parts[1]) : 0.90;
      source = 'Hardware Serial';
    }

    // Check query params as fallback (e.g. GET/POST /api/predict?coin=5rup)
    if (!coin && req.query.coin) coin = req.query.coin;
    if (!confidence && req.query.confidence) confidence = parseFloat(req.query.confidence);
    if (!sensorValue && req.query.sensorValue) sensorValue = req.query.sensorValue;

    // Default coin fallback if only raw sensor value was sent
    if (!coin && sensorValue) {
      coin = '5rup'; // default prediction
    }

    if (!coin && !sensorValue) {
      return res.status(400).json({
        error: 'Missing required parameters. Send {"coin": "5rup", "confidence": 0.92} or {"sensorValue": 750}'
      });
    }

    const prediction = await processPrediction({
      coin,
      confidence,
      sensorValue,
      source,
      isSimulated: false
    });

    console.log(`[HARDWARE INGEST] ${prediction.source} -> ${prediction.coin} (${Math.round(prediction.confidence * 100)}%) -> Malayalam Quote: "${prediction.malayalamText}"`);
    return res.json({ success: true, event: prediction });
  } catch (err) {
    console.error('[HARDWARE INGEST ERROR]', err.message);
    return res.status(400).json({ error: err.message });
  }
};

app.post('/api/predict', handleHardwareIngest);
app.post('/api/sensor', handleHardwareIngest);
app.get('/api/predict', handleHardwareIngest);

// POST /api/simulate
app.post('/api/simulate', async (req, res) => {
  try {
    const coinKeys = Object.keys(COIN_MAP);
    const requestedCoin = req.body && req.body.coin;
    const selectedCoin = (requestedCoin && normalizeCoinLabel(requestedCoin)) ? normalizeCoinLabel(requestedCoin) : getRandomItem(coinKeys);

    // Simulated realistic confidence between 0.75 and 0.98
    const simulatedConfidence = parseFloat((Math.random() * (0.98 - 0.75) + 0.75).toFixed(2));

    const prediction = await processPrediction({
      coin: selectedCoin,
      confidence: simulatedConfidence,
      source: 'Simulator',
      isSimulated: true
    });

    return res.json(prediction);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
});

// Native TCP Socket Server (For raw TCP Ethernet socket senders)
const tcpServer = net.createServer((socket) => {
  const remoteIp = socket.remoteAddress;
  console.log(`\n=================================`);
  console.log(`[TCP ETHERNET CONNECTED] Sender: ${remoteIp}`);
  console.log(`=================================`);

  socket.on('data', async (data) => {
    try {
      const rawText = data.toString('utf-8').trim();
      console.log(`[RAW TCP RECEIVED]: ${JSON.stringify(rawText)}`);

      if (rawText) {
        // Parse coin prediction string e.g. "5rup", "5", "5rup,0.95"
        const parts = rawText.split(',');
        const coin = parts[0];
        const confidence = parts[1] ? parseFloat(parts[1]) : 0.92;

        const prediction = await processPrediction({
          coin,
          confidence,
          source: `Teammate ML (TCP Ethernet)`
        });

        console.log(`[TCP PREDICTION PROCESSED] -> ${prediction.coin} (${Math.round(prediction.confidence * 100)}%) -> Malayalam: "${prediction.malayalamText}"`);
        socket.write(`OK: ${prediction.coin}\n`);
      }
    } catch (err) {
      console.error('[TCP PARSE ERROR]', err.message);
    }
  });

  socket.on('close', () => {
    console.log(`[TCP ETHERNET] Connection closed by ${remoteIp}`);
  });

  socket.on('error', (err) => {
    console.error('[TCP SOCKET ERROR]', err.message);
  });
});

tcpServer.listen(TCP_PORT, () => {
  console.log(`TCP Socket Receiver: tcp://0.0.0.0:${TCP_PORT}`);
});

// Start HTTP & WebSocket Server
server.listen(PORT, () => {
  console.log(`=================================`);
  console.log(`VIBRACOIN Backend Running`);
  console.log(`HTTP Server: http://localhost:${PORT}`);
  console.log(`WebSocket Server: ws://localhost:${PORT}`);
  console.log(`TCP Receiver: tcp://localhost:${TCP_PORT}`);
  console.log(`=================================`);
});

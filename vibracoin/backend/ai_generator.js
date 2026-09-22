const http = require('http');
const https = require('https');
require('dotenv').config();

// Fallback curated Malayalam responses per coin denomination
const MALAYALAM_FALLBACKS = {
  '1rup': [
    { ml: "ഒരു രൂപയോ? തുടക്കം ചെറുതാണെങ്കിലും ലക്ഷ്യം വലുതാണ്!", en: "One rupee? Small start, but big goals!" },
    { ml: "ദാ വന്നു ഒരു രൂപ! പ്ലേറ്റ് ചെറുതായി ഒന്നനങ്ങി.", en: "Here comes 1 rupee! The plate barely moved." },
    { ml: "ചെറിയ നാണയം, പക്ഷെ ഗമയ്ക്ക് കുറവൊന്നുമില്ല!", en: "Small coin, but no shortage of attitude!" }
  ],
  '2rup': [
    { ml: "രണ്ടു രൂപ എത്തിപ്പോയി! കൊള്ളാം, ശബ്ദം കേട്ടു.", en: "Two rupees has arrived! Nice, heard that sound." },
    { ml: "രണ്ടു രൂപയുടെ വീഴ്ച... പ്ലേറ്റ് ഹരത്തിലായി!", en: "Two rupee drop... the plate got excited!" },
    { ml: "ഇരട്ട രൂപയുടെ ഗരിമ! സുന്ദരമായ വീഴ്ച.", en: "Dignity of two rupees! Beautiful drop." }
  ],
  '5rup': [
    { ml: "അമ്പോ 5 രൂപ! പ്ലേറ്റ് കുലുങ്ങിപ്പോയി മക്കളെ!", en: "Wow ₹5! The plate shook completely, folks!" },
    { ml: "കനത്ത നാണയം വന്നു! ഇതാണ് യഥാർത്ഥ മാസ് എൻട്രി!", en: "Heavy coin arrived! Now that is a mass entry!" },
    { ml: "അഞ്ച് രൂപയുടെ ഗാംഭീര്യം! പ്ലേറ്റ് വീണു നമിച്ചു.", en: "Majesty of 5 rupees! The plate bowed down." }
  ],
  '10rup': [
    { ml: "പത്തു രൂപയുടെ വീഴ്ച! താളം കിടുക്കി!", en: "Ten rupee drop! Rhythm was awesome!" },
    { ml: "ദാ കിടക്കുന്നു 10 രൂപ! നല്ല പവറുള്ള എൻട്രി!", en: "There lies ₹10! Powerful entry!" },
    { ml: "പത്തു രൂപ വീണ ശബ്ദം കേട്ട് പ്ലേറ്റ് ഞെട്ടി!", en: "The plate was shocked hearing the ₹10 sound!" }
  ],
  '20rup': [
    { ml: "ഇരുപതു രൂപ! സംഗതി ജോറായി, രാജകീയ എൻട്രി!", en: "Twenty rupees! Grand royal entry!" },
    { ml: "അതിമാരക വീഴ്ച! 20 രൂപയുടെ മാസ്സ് സംഭവം!", en: "Lethal drop! ₹20 mass event!" },
    { ml: "പ്ലേറ്റിന്റെ മൂല്യം കൂടി! 20 രൂപ എത്തിയിട്ടുണ്ട്!", en: "Plate value increased! ₹20 has arrived!" }
  ]
};

function getRandomFallback(coinLabel) {
  const list = MALAYALAM_FALLBACKS[coinLabel] || MALAYALAM_FALLBACKS['5rup'];
  return list[Math.floor(Math.random() * list.length)];
}

/**
 * Call local Ollama API (Liquid LFM / Llama)
 */
async function generateViaOllama(coinName, confidence) {
  const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
  const model = process.env.OLLAMA_MODEL || 'liquid/lfm';

  const prompt = `Generate a single short, witty, funny 1-sentence reaction in Malayalam language (Malayalam script) for a coin impact event. Coin: ${coinName}, Confidence: ${Math.round(confidence * 100)}%. Also provide the English translation. Output MUST be valid JSON: {"malayalamText": "...", "englishText": "..."}`;

  return new Promise((resolve, reject) => {
    try {
      const url = new URL(`${ollamaHost}/api/generate`);
      const payload = JSON.stringify({
        model,
        prompt,
        stream: false,
        format: 'json'
      });

      const options = {
        hostname: url.hostname,
        port: url.port || 11434,
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        },
        timeout: 4000
      };

      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            if (res.statusCode === 200) {
              const parsed = JSON.parse(data);
              const responseText = parsed.response;
              const jsonResult = JSON.parse(responseText);
              if (jsonResult.malayalamText) {
                return resolve(jsonResult);
              }
            }
            reject(new Error(`Ollama status ${res.statusCode}`));
          } catch (e) {
            reject(e);
          }
        });
      });

      req.on('error', (err) => reject(err));
      req.on('timeout', () => { req.destroy(); reject(new Error('Ollama timeout')); });
      req.write(payload);
      req.end();
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Call Gemini API if API key is present in environment
 */
async function generateViaGemini(coinName, confidence) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('No GEMINI_API_KEY defined');

  const prompt = `Generate a single short, witty, funny 1-sentence reaction in Malayalam language (Malayalam script) for a coin impact event. Coin: ${coinName}, Confidence: ${Math.round(confidence * 100)}%. Also provide the English translation. Output format strict JSON: {"malayalamText": "...", "englishText": "..."}`;

  const payload = JSON.stringify({
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: { responseMimeType: "application/json" }
  });

  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'generativelanguage.googleapis.com',
      path: `/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      },
      timeout: 4000
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          if (res.statusCode === 200) {
            const parsed = JSON.parse(data);
            const textStr = parsed.candidates[0].content.parts[0].text;
            const jsonResult = JSON.parse(textStr);
            if (jsonResult.malayalamText) {
              return resolve(jsonResult);
            }
          }
          reject(new Error(`Gemini status ${res.statusCode}`));
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', (err) => reject(err));
    req.on('timeout', () => { req.destroy(); reject(new Error('Gemini timeout')); });
    req.write(payload);
    req.end();
  });
}

/**
 * Main AI Malayalam Quote Generator with Fallbacks:
 * 1. Ollama (Liquid/LFM)
 * 2. Gemini API
 * 3. Curated Malayalam Fallback Catalog
 */
async function generateMalayalamResponse(coinLabel, confidence = 0.90) {
  const coinNameMap = { '1rup': '₹1', '2rup': '₹2', '5rup': '₹5', '10rup': '₹10', '20rup': '₹20' };
  const coinName = coinNameMap[coinLabel] || coinLabel;

  // Try 1: Ollama (Liquid LFM)
  try {
    const res = await generateViaOllama(coinName, confidence);
    console.log('[AI GENERATOR] Generated via Ollama (Liquid/LFM)');
    return { ...res, source: 'Liquid/LFM (Ollama)' };
  } catch (err) {
    // console.log('[AI GENERATOR] Ollama unavailable, trying Gemini API...');
  }

  // Try 2: Gemini API
  try {
    const res = await generateViaGemini(coinName, confidence);
    console.log('[AI GENERATOR] Generated via Gemini API');
    return { ...res, source: 'Gemini API' };
  } catch (err) {
    // console.log('[AI GENERATOR] Gemini API unavailable, using Malayalam fallback catalog.');
  }

  // Try 3: Curated Malayalam Fallback Catalog
  const fallback = getRandomFallback(coinLabel);
  return {
    malayalamText: fallback.ml,
    englishText: fallback.en,
    source: 'VibraCoin ML Catalog'
  };
}

module.exports = {
  generateMalayalamResponse
};

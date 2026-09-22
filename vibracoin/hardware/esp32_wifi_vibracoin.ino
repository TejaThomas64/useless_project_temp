/*
  ==============================================================
  VIBRACOIN - ESP32 Wi-Fi Hardware Firmware
  ==============================================================
  Sends vibration sensor detection directly to Node.js backend over Wi-Fi.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server API URL (Use your computer's local IP address, e.g. 192.168.1.X)
const char* serverUrl = "http://192.168.1.100:5000/api/predict";

// Sensor Configuration
const int SENSOR_PIN = 34; // Piezoelectric sensor analog input
const int THRESHOLD = 500; // Trigger threshold

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected! IP Address: ");
  Serial.println(WiFi.localIP());
}

void sendDetectionToBackend(String coinLabel, float confidenceScore, int rawSensorVal) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<250> doc;
    doc["coin"] = coinLabel;          // '1rup', '2rup', '5rup', '10rup', '20rup'
    doc["confidence"] = confidenceScore;
    doc["sensorValue"] = rawSensorVal;
    doc["source"] = "ESP32 Wi-Fi";

    String jsonBuffer;
    serializeJson(doc, jsonBuffer);

    int httpCode = http.POST(jsonBuffer);
    if (httpCode > 0) {
      Serial.printf("[SUCCESS] Sent coin hit! Response Code: %d\n", httpCode);
    } else {
      Serial.printf("[ERROR] HTTP POST failed: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }
}

void loop() {
  int rawVal = analogRead(SENSOR_PIN);

  // Detect vibration spike
  if (rawVal > THRESHOLD) {
    Serial.printf("[IMPACT DETECTED] Peak ADC Value: %d\n", rawVal);

    // Replace with your ML model prediction logic or threshold mapping
    String coinResult = "5rup"; // Options: '1rup', '2rup', '5rup', '10rup', '20rup'
    float confidence = 0.94;

    sendDetectionToBackend(coinResult, confidence, rawVal);

    // Debounce delay to avoid duplicate triggers from vibrating plate
    delay(1800);
  }
}

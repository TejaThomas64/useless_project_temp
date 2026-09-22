/*
  ==============================================================
  VIBRACOIN - ESP32 + MPU6050 Accelerometer Wi-Fi Firmware
  ==============================================================
  Reads 3-axis Acceleration (ax, ay, az) from MPU6050 over I2C.
  Detects coin impact spike and POSTs sensor data to backend.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

// Wi-Fi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server API URL (Use your Laptop's LAN IP address, e.g. http://192.168.1.100:5000/api/sensor)
const char* serverUrl = "http://192.168.1.100:5000/api/sensor";

// MPU6050 Object
Adafruit_MPU6050 mpu;

// Vibration threshold magnitude (m/s^2 above baseline 9.8)
const float IMPACT_THRESHOLD = 3.5; 

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // I2C SDA = GPIO21, SCL = GPIO22

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("[ERROR] Could not find MPU6050 chip!");
    while (1) { delay(10); }
  }
  Serial.println("[OK] MPU6050 Accelerometer Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);

  // Connect Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[OK] Wi-Fi Connected! IP: ");
  Serial.println(WiFi.localIP());
}

void sendSensorDataToBackend(float ax, float ay, float az, float peakMag) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<300> doc;
    // Sending both raw acceleration sample & peak magnitude metadata
    doc["ax"] = ax;
    doc["ay"] = ay;
    doc["az"] = az;
    doc["sensorValue"] = peakMag * 100;
    doc["source"] = "ESP32 MPU6050";

    String jsonPayload;
    serializeJson(doc, jsonPayload);

    int httpResponseCode = http.POST(jsonPayload);
    if (httpResponseCode > 0) {
      Serial.printf("[SUCCESS] MPU6050 data posted! HTTP Status: %d\n", httpResponseCode);
    } else {
      Serial.printf("[ERROR] HTTP POST failed: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  }
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Compute Signal Vector Magnitude (SVM)
  float magnitude = sqrt(a.acceleration.x * a.acceleration.x + 
                         a.acceleration.y * a.acceleration.y + 
                         a.acceleration.z * a.acceleration.z);

  // Check for impact spike relative to gravity (9.8 m/s^2)
  float deltaMag = abs(magnitude - 9.81);

  if (deltaMag > IMPACT_THRESHOLD) {
    Serial.printf("[IMPACT DETECTED] Peak Delta Magnitude: %.2f m/s^2\n", deltaMag);

    sendSensorDataToBackend(a.acceleration.x, a.acceleration.y, a.acceleration.z, deltaMag);

    // Debounce to avoid duplicate counts from residual vibrating plate
    delay(1800);
  }

  delay(10); // 100Hz sampling loop
}

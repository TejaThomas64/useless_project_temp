/*
  ==============================================================
  VIBRACOIN - ESP32 / Arduino Ethernet (W5500 / LAN8720) Firmware
  ==============================================================
  Reads MPU6050 accelerometer & sends sensor data to backend over wired Ethernet.
*/

#include <SPI.h>
#include <Ethernet.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// MAC Address for Ethernet Shield (Must be unique on local network)
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };

// Laptop LAN IP running Node.js backend
IPAddress server(192, 168, 1, 100); 
const int port = 5000;

EthernetClient client;
Adafruit_MPU6050 mpu;

const float IMPACT_THRESHOLD = 3.5; // m/s^2 above baseline 9.8

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // I2C SDA = GPIO21, SCL = GPIO22

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("[ERROR] MPU6050 not found!");
    while (1) delay(10);
  }
  Serial.println("[OK] MPU6050 Initialized!");

  // Start Ethernet connection via DHCP
  Serial.println("Initializing Ethernet via DHCP...");
  if (Ethernet.begin(mac) == 0) {
    Serial.println("[ERROR] Failed to configure Ethernet using DHCP");
    // Fallback static IP if DHCP fails
    IPAddress ip(192, 168, 1, 150);
    Ethernet.begin(mac, ip);
  }
  
  delay(1000);
  Serial.print("[OK] Ethernet Connected! IP Address: ");
  Serial.println(Ethernet.localIP());
}

void sendEthernetPost(String jsonPayload) {
  if (client.connect(server, port)) {
    Serial.println("Sending HTTP POST over Ethernet...");
    client.println("POST /api/sensor HTTP/1.1");
    client.print("Host: "); client.println(server);
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(jsonPayload.length());
    client.println();
    client.println(jsonPayload);

    // Read response acknowledgment
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        // Serial.print(c);
      }
    }
    client.stop();
    Serial.println("[OK] Ethernet POST complete!");
  } else {
    Serial.println("[ERROR] Connection to server failed!");
  }
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float magnitude = sqrt(a.acceleration.x * a.acceleration.x + 
                         a.acceleration.y * a.acceleration.y + 
                         a.acceleration.z * a.acceleration.z);

  float deltaMag = abs(magnitude - 9.81);

  if (deltaMag > IMPACT_THRESHOLD) {
    Serial.printf("[IMPACT DETECTED] Peak Delta: %.2f m/s^2\n", deltaMag);

    String payload = "{\"ax\":" + String(a.acceleration.x) + 
                     ",\"ay\":" + String(a.acceleration.y) + 
                     ",\"az\":" + String(a.acceleration.z) + 
                     ",\"sensorValue\":" + String((int)(deltaMag * 100)) + 
                     ",\"source\":\"Ethernet MPU6050\"}";

    sendEthernetPost(payload);
    delay(1800); // Debounce
  }

  delay(10);
}

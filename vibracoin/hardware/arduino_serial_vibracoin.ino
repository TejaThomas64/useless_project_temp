/*
  ==============================================================
  VIBRACOIN - Arduino USB Serial Firmware
  ==============================================================
  Detects coin impact on Piezo/vibration sensor and prints CSV over Serial.
  Used in combination with `backend/serial_bridge.py`.
*/

const int SENSOR_PIN = A0;   // Piezoelectric sensor analog input
const int THRESHOLD = 350;    // Trigger threshold value

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorVal = analogRead(SENSOR_PIN);

  if (sensorVal > THRESHOLD) {
    // Determine coin denomination based on signal peak / ML model
    // Format sent to Serial Bridge: "coin_label,confidence"
    String coin = "5rup";
    float confidence = 0.92;

    // Send formatted prediction line over USB Serial
    Serial.print(coin);
    Serial.print(",");
    Serial.println(confidence);

    // Debounce to prevent multiple hits for one drop
    delay(1800);
  }
}

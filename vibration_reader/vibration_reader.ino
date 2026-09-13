#include <Wire.h>

#define MPU_ADDR 0x68

// Sampling frequency
const unsigned long SAMPLE_INTERVAL_US = 5000;  // 200 Hz

unsigned long nextSampleTime = 0;


// --------------------------------------------------
// SETUP
// --------------------------------------------------

void setup() {

  Serial.begin(115200);

  delay(1000);

  Wire.begin(21, 22);

  setupMPU();

  Serial.println("FFT_DATA_START");

  nextSampleTime = micros();
}


// --------------------------------------------------
// MAIN LOOP
// --------------------------------------------------

void loop() {

  unsigned long currentTime = micros();

  if ((long)(currentTime - nextSampleTime) >= 0) {

    nextSampleTime += SAMPLE_INTERVAL_US;

    float x, y, z;

    readAcceleration(x, y, z);

    // Timestamp in microseconds
    Serial.print(currentTime);
    Serial.print(",");

    Serial.print(x, 6);
    Serial.print(",");

    Serial.print(y, 6);
    Serial.print(",");

    Serial.println(z, 6);
  }
}


// --------------------------------------------------
// MPU6050 SETUP
// --------------------------------------------------

void setupMPU() {

  // Wake MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {

    Serial.println("ERROR: MPU6050 NOT FOUND");

    while (1) {
      delay(1000);
    }
  }

  // Accelerometer ±2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();

  // Gyroscope ±250 degrees/sec
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission();
}


// --------------------------------------------------
// READ ACCELERATION
// --------------------------------------------------

void readAcceleration(float &x, float &y, float &z) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);

  // Keep I2C connection active for read
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, 6);

  if (Wire.available() < 6) {

    x = 0;
    y = 0;
    z = 0;

    return;
  }

  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();

  x = rawX / 16384.0;
  y = rawY / 16384.0;
  z = rawZ / 16384.0;
}

#include <Wire.h>

#define MPU_ADDR 0x68

// 5000 microseconds = 200 Hz nominal sampling rate
const unsigned long SAMPLE_INTERVAL_US = 5000;

unsigned long nextSampleTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // ESP32 I2C pins
  Wire.begin(21, 22);

  setupMPU();

  Serial.println("FFT_DATA_START");

  nextSampleTime = micros();
}

void loop() {

  unsigned long currentTime = micros();

  if ((long)(currentTime - nextSampleTime) >= 0) {

    nextSampleTime += SAMPLE_INTERVAL_US;

    float x, y, z;

    readAcceleration(x, y, z);

    // Output:
    // timestamp_us,x_g,y_g,z_g

    Serial.print(currentTime);
    Serial.print(",");

    Serial.print(x, 6);
    Serial.print(",");

    Serial.print(y, 6);
    Serial.print(",");

    Serial.println(z, 6);
  }
}


void setupMPU() {

  // --------------------------------------------------
  // Wake up MPU6050
  // --------------------------------------------------

  Wire.beginTransmission(MPU_ADDR);

  // Power management register
  Wire.write(0x6B);

  // Wake up
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {

    Serial.println("ERROR: MPU6050 NOT FOUND");

    while (1) {
      delay(1000);
    }
  }


  // --------------------------------------------------
  // Configure accelerometer
  // --------------------------------------------------

  Wire.beginTransmission(MPU_ADDR);

  // Accelerometer configuration register
  Wire.write(0x1C);

  // 0x08 = ±4g
  Wire.write(0x08);

  if (Wire.endTransmission() != 0) {

    Serial.println("ERROR: ACCELEROMETER CONFIG FAILED");

    while (1) {
      delay(1000);
    }
  }


  // --------------------------------------------------
  // Configure gyroscope
  // --------------------------------------------------

  Wire.beginTransmission(MPU_ADDR);

  // Gyroscope configuration register
  Wire.write(0x1B);

  // 0x00 = ±250 degrees/second
  Wire.write(0x00);

  Wire.endTransmission();
}


void readAcceleration(float &x, float &y, float &z) {

  // Start reading at accelerometer X register
  Wire.beginTransmission(MPU_ADDR);

  Wire.write(0x3B);

  // Repeated start
  Wire.endTransmission(false);

  // Request 6 bytes:
  // X high, X low
  // Y high, Y low
  // Z high, Z low

  Wire.requestFrom(MPU_ADDR, 6);

  if (Wire.available() < 6) {

    x = 0;
    y = 0;
    z = 0;

    return;
  }


  // Combine high and low bytes
  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();


  // --------------------------------------------------
  // ±4g sensitivity
  //
  // 8192 LSB = 1g
  // --------------------------------------------------

  x = rawX / 8192.0;
  y = rawY / 8192.0;
  z = rawZ / 8192.0;
}

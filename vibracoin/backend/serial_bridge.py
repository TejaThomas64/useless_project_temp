import serial
import time
import requests
import sys

# Configure your microcontroller Serial Port
# Windows: 'COM3', 'COM4', etc.
# Linux / Raspberry Pi: '/dev/ttyUSB0' or '/dev/ttyACM0'
# Mac: '/dev/cu.usbmodem14101'
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
API_URL = 'http://localhost:5000/api/predict'

def run_bridge():
    print("==================================================")
    print("  VIBRACOIN Hardware Serial Bridge")
    print(f"  Listening on Port: {SERIAL_PORT} @ {BAUD_RATE} baud")
    print(f"  Target Server API: {API_URL}")
    print("==================================================")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for serial port connection stabilization
        print(" Connected! Waiting for sensor signals from microcontroller...\n")
    except Exception as e:
        print(f" Error connecting to serial port '{SERIAL_PORT}': {e}")
        print(" Please check your COM port name and ensure Arduino IDE Serial Monitor is closed.")
        sys.exit(1)

    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[MC SIGNAL RECEIVED]: {line}")
                
                # Expected MCU output formats:
                # 1) "5rup,0.92" or "5rup" (coin label, optional confidence)
                # 2) "750" (raw ADC sensor reading)
                parts = line.split(',')
                coin = parts[0]
                confidence = float(parts[1]) if len(parts) > 1 else 0.90
                
                payload = {
                    "coin": coin,
                    "confidence": confidence,
                    "source": "Arduino Serial USB"
                }

                # Send data to Node.js backend
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    print(f" Sent to web server successfully -> Broadcasted to UI!")
                else:
                    print(f" Server response error: {res.status_code} - {res.text}")

        except KeyboardInterrupt:
            print("\nExiting bridge...")
            break
        except Exception as e:
            print(f" Error processing line: {e}")

if __name__ == '__main__':
    run_bridge()

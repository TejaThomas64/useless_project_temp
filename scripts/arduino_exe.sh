#!/bin/zsh

echo "=============================="
echo "  Compiling vibration_reader"
echo "=============================="

arduino-cli compile --fqbn esp32:esp32:esp32 ~/vibration_reader

if [ $? -ne 0 ]; then
    echo ""
    echo "!!! COMPILE FAILED !!!"
    exit 1
fi

echo ""
echo "=============================="
echo "  Uploading to ESP32"
echo "=============================="

arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 ~/vibration_reader

if [ $? -ne 0 ]; then
    echo ""
    echo "!!! UPLOAD FAILED !!!"
    exit 1
fi

echo ""
echo "=============================="
echo "  Upload successful"
echo "  Starting serial monitor"
echo "=============================="
echo ""

arduino-cli monitor -p /dev/ttyUSB0 -c baudrate=115200

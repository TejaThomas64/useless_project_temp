<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# Vibracoin 🎯


## Basic Details
### Team Name: Double T


 ### Team Members
- Team Lead: Teja Thomas - Rajiv Gandhi Institute of Technology,Kottayam
- Member 2: Nathan George Panathara - Rajiv Gandhi Institute of Technology, Kottayam


### Project Description
Our project detects the value of a coin mainly 1,2,5,10 and 20 rupees, using an MPU6050 and an ESP32 microcontroller.The ESP32 sends the vibrations to one laptop which uses an ML model to predict the value, and sends it to another laptop which displays the value and a message making fun of you on a website with the help of Gemini API.

### The Problem (that doesn't exist)
There isn't a reliable solution for blind people to know the value of their coins(OF course they can ask someone else or touch the coins, but its needs to be AI-POWEREDDD).
### The Solution (that nobody asked for)
VIBRACOIN with its high-tech sophistication, analyses your coin's signature , passes the oscillations through a physics based hierarchial ML model to reveal the value with a twist, ridiculing that stingy prick in nadan yet polite way.
[How are you solving it? Keep it fun!]
We use an MPU6050 to detect vibrations detected by the plate and turn it into a readable format with an ESP32.This data is then passed through an ML model which uses a hierarchial architecture to deduce to coins value.Accuracy is mixed but still accurate if there is only few coins in the plate.We then pass this to another laptop using internet(TCP-IP).

Technical Details

Hardware

- MPU6050 accelerometer/gyroscope captures vibration and acceleration produced when a coin impacts the sensing plate.
- ESP32 collects the sensor data and communicates with the software system over Wi-Fi.
- A physical plate acts as the vibration-transmitting surface.

Signal Processing & ML

- Raw X, Y and Z-axis acceleration data is collected at approximately 200 Hz.
- The vibration signal is filtered and segmented into individual impact events.
- Time-domain and frequency-domain features are extracted from each event, including amplitude, RMS, energy, peak behaviour, FFT-based features and post-impact/ring-down characteristics.
- A machine-learning classifier predicts the denomination:
  ₹1, ₹2, ₹5, ₹10 or ₹20.
- The model produces a probability/confidence score along with the predicted denomination.

AI Pipeline

- The ML prediction is passed to a locally running Liquid/LFM model through Ollama.
- Liquid interprets the numerical prediction and impact characteristics.
- The interpreted result is passed to the Gemini API, which generates a short, quirky Malayalam reaction.
- A fallback mechanism allows the system to continue functioning if the local model or Gemini API is unavailable.

Backend

- Node.js + Express handles communication between the hardware/ML system and the web application.
- REST API endpoints receive predictions from the ESP32/ML pipeline.
- WebSockets provide real-time updates to the frontend without polling.

Frontend

- React + Vite provides a live visualization dashboard.
- Displays:
  - Detected denomination
  - ML confidence/probability
  - Impact strength
  - Vibration visualization
  - Recent detections
  - AI-generated Malayalam reaction
  - Connection/model status
- Browser-based Malayalam text-to-speech is used to speak the generated reaction.

Overall Pipeline

MPU6050 → ESP32 → ML Model → Liquid/LFM → Gemini API → Malayalam TTS → React Dashboard

The system combines vibration sensing, signal processing, machine learning, local LLM inference, cloud generative AI and real-time web technologies into a single coin-detection experience.

### Implementation

# Run
[commands]

### Project Documentation

# Diagrams
![Workflow](Add your workflow/architecture diagram here)
*Add caption explaining your workflow*

For Hardware:

# Schematic & Circuit
<img width="607" height="496" alt="WhatsApp Image 2026-09-12 at 7 01 22 AM" src="https://github.com/user-attachments/assets/90b6d9e5-6b59-45f9-9482-bcdf1e87df5e" />



MPU6050 and ESP32 connection.ESP32 is connected tot eh computer via USB.

# Build Photos
Components
<img width="515" height="388" alt="images" src="https://github.com/user-attachments/assets/f2d88fa1-4ee3-4bfc-9816-b6142ce1144c" />

<img width="302" height="600" alt="image" src="https://github.com/user-attachments/assets/2ce5fc91-6e0f-4d83-9821-618b58d6474c" />

MPU6050 → ESP32


![Final](Add photo of final product here)
*Explain the final build*
<img width="1904" height="853" alt="Screenshot 2026-09-12 063313" src="https://github.com/user-attachments/assets/73a3af7b-60c9-4d8f-a8ed-427f83c2ab7c" />

### Project Demo
# Video
https://drive.google.com/file/d/1kxt9m2ZNvLMUYOdukarecUf_LKHvZxtk/view?usp=drivesdk



## Team Contributions
- Teja Thomas: Gemini API and local Liquid AI modal
- Nathan George: Local ML model predicting coin values and esp32 , mpu6050 vibration reader

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)




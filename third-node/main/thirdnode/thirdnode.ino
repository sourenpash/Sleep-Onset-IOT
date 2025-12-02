#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <math.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_ADS1X15.h>          
#include "Adafruit_BME680.h"           

// BLE
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";

const char* SERVER_IP     = "10.0.0.32";
const uint16_t SERVER_PORT = 5000;

const char* NODE_ID       = "door";        
const char* NODE_BLE_NAME = "door-node";   

const char* PEER1_NAME    = "window-node";
const char* PEER2_NAME    = "bed-node";

WiFiClient client;

const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
unsigned long lastFeatureMillis  = 0;

String rxBuffer;

Adafruit_BME680 bme;
Adafruit_ADS1015 ads;

bool bme_ok  = false;
bool ads_ok  = false;

const int MIC_SAMPLES = 40;

BLEScan*        pBLEScan      = nullptr;
BLEAdvertising* pAdvertising  = nullptr;
const int BLE_SCAN_TIME_SEC   = 2;
const int RSSI_INVALID        = -999;

const unsigned long BLE_SCAN_INTERVAL_MS = 5UL * 60UL * 1000UL;
unsigned long lastBleScanMillis = 0;   

int lastPeer1Rssi = RSSI_INVALID;
int lastPeer2Rssi = RSSI_INVALID;

struct DoorSample {
  float temp_c;
  float hum_pct;
  float mic_v;
  float light_v;
};

struct Kalman1D {
  float x;          // state estimate
  float P;          // estimate covariance
  float Q;          // process noise
  float R;          // measurement noise
  bool  initialized;

  Kalman1D(float q = 0.01f, float r = 1.0f)
    : x(0.0f), P(1.0f), Q(q), R(r), initialized(false) {}

  float update(float z) {
    if (isnan(z)) {
      return initialized ? x : NAN;
    }

    if (!initialized) {
      x = z;
      P = 1.0f;
      initialized = true;
      return x;
    }

    P += Q;

    float K = P / (P + R);
    x = x + K * (z - x);
    P = (1.0f - K) * P;

    return x;
  }
};

Kalman1D kf_temp(0.01f, 0.25f);   // temperature: low noise
Kalman1D kf_hum(0.01f, 1.0f);     // humidity
Kalman1D kf_mic(0.1f, 10.0f);     // mic voltage (noisy)
Kalman1D kf_light(0.05f, 5.0f);   // light voltage


// Temperature at door (°C) from BME680
float CAL_TEMP_DOOR_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };   
float CAL_TEMP_DOOR_RAW[3]    = { 0.0f, 0.0f, 0.0f };   

// Humidity at door (%) from BME680
float CAL_HUM_DOOR_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_HUM_DOOR_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Microphone voltage (V) from ADS1015 CH0
float CAL_MIC_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_MIC_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Light voltage (V) from ADS1015 CH2
float CAL_LIGHT_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_LIGHT_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Calibration params: actual ≈ a * raw + b
float calTempDoor_a = 1.0f, calTempDoor_b = 0.0f; bool calTempDoor_enabled = false;
float calHumDoor_a  = 1.0f, calHumDoor_b  = 0.0f; bool calHumDoor_enabled  = false;
float calMic_a      = 1.0f, calMic_b      = 0.0f; bool calMic_enabled      = false;
float calLight_a    = 1.0f, calLight_b    = 0.0f; bool calLight_enabled    = false;

// Generic least-squares line y = a x + b over up to n points
// (actual[i], raw[i]) with (0,0) treated as "unused".
void computeCalGeneric(const float actual[], const float raw[], int n,
                       float &a, float &b, bool &enabled) {
  float sumX = 0.0f, sumY = 0.0f, sumXX = 0.0f, sumXY = 0.0f;
  int count = 0;

  for (int i = 0; i < n; ++i) {
    float ya = actual[i];
    float xr = raw[i];

    if (ya == 0.0f && xr == 0.0f) {
      continue;
    }
    if (isnan(ya) || isnan(xr)) {
      continue;
    }

    sumX  += xr;
    sumY  += ya;
    sumXX += xr * xr;
    sumXY += xr * ya;
    count++;
  }

  if (count == 0) {
    a = 1.0f;
    b = 0.0f;
    enabled = false;
    return;
  }

  float denom = (count * sumXX - sumX * sumX);
  if (denom == 0.0f) {
    a = 1.0f;
    b = 0.0f;
    enabled = false;
    return;
  }

  a = (count * sumXY - sumX * sumY) / denom;
  b = (sumY - a * sumX) / count;
  enabled = true;
}

float applyCal(float raw, float a, float b, bool enabled) {
  if (!enabled || isnan(raw)) return raw;
  return a * raw + b;
}

void initCalibration() {
  computeCalGeneric(CAL_TEMP_DOOR_ACTUAL, CAL_TEMP_DOOR_RAW, 3,
                    calTempDoor_a, calTempDoor_b, calTempDoor_enabled);
  computeCalGeneric(CAL_HUM_DOOR_ACTUAL,  CAL_HUM_DOOR_RAW,  3,
                    calHumDoor_a,  calHumDoor_b,  calHumDoor_enabled);
  computeCalGeneric(CAL_MIC_ACTUAL,       CAL_MIC_RAW,       3,
                    calMic_a,      calMic_b,      calMic_enabled);
  computeCalGeneric(CAL_LIGHT_ACTUAL,     CAL_LIGHT_RAW,     3,
                    calLight_a,    calLight_b,    calLight_enabled);

  Serial.println("[DOOR] Calibration init:");
  Serial.print("  Temp_door: ");
  Serial.print(calTempDoor_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calTempDoor_a, 4);
  Serial.print(" b="); Serial.println(calTempDoor_b, 4);

  Serial.print("  Hum_door:  ");
  Serial.print(calHumDoor_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calHumDoor_a, 4);
  Serial.print(" b="); Serial.println(calHumDoor_b, 4);

  Serial.print("  Mic_v:     ");
  Serial.print(calMic_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calMic_a, 4);
  Serial.print(" b="); Serial.println(calMic_b, 4);

  Serial.print("  Light_v:   ");
  Serial.print(calLight_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calLight_a, 4);
  Serial.print(" b="); Serial.println(calLight_b, 4);
}

void logLine(const String& msg) {
  Serial.println(msg);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  logLine("[DOOR] Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  logLine("[DOOR] WiFi connected");
  Serial.print("[DOOR] IP: ");
  Serial.println(WiFi.localIP());
}

bool connectToServer() {
  logLine("[DOOR] Connecting to server...");
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    logLine("[DOOR] Failed to connect to server");
    return false;
  }
  logLine("[DOOR] Connected to server");
  return true;
}

void sendHello() {
  String hello = "{\"type\":\"hello\",\"node\":\"";
  hello += NODE_ID;
  hello += "\"}\n";
  client.print(hello);
  logLine("[DOOR] Sent hello: " + hello);
}

void initBLE() {
  logLine("[DOOR] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);

  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[DOOR] BLE advertising started as: " + String(NODE_BLE_NAME));

  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(80);
  logLine("[DOOR] BLE scanner ready");
}

void scanForPeers(int& peer1Rssi, int& peer2Rssi) {
  peer1Rssi = RSSI_INVALID;
  peer2Rssi = RSSI_INVALID;

  if (!pBLEScan) return;

  BLEScanResults* results = pBLEScan->start(BLE_SCAN_TIME_SEC, false);
  if (!results) {
    pBLEScan->clearResults();
    logLine("[DOOR] BLE scan returned null results");
    return;
  }

  int count = results->getCount();

  for (int i = 0; i < count; ++i) {
    BLEAdvertisedDevice dev = results->getDevice(i);
    String devName = String(dev.getName().c_str());
    int rssi = dev.getRSSI();

    if (devName == PEER1_NAME) {
      if (peer1Rssi == RSSI_INVALID || rssi > peer1Rssi) {
        peer1Rssi = rssi;
      }
    } else if (devName == PEER2_NAME) {
      if (peer2Rssi == RSSI_INVALID || rssi > peer2Rssi) {
        peer2Rssi = rssi;
      }
    }
  }

  pBLEScan->clearResults();

  if (peer1Rssi == RSSI_INVALID) {
    logLine("[DOOR] BLE peer1 (" + String(PEER1_NAME) + ") not seen this interval");
  } else {
    logLine("[DOOR] BLE peer1 (" + String(PEER1_NAME) + ") RSSI = " + String(peer1Rssi));
  }
  if (peer2Rssi == RSSI_INVALID) {
    logLine("[DOOR] BLE peer2 (" + String(PEER2_NAME) + ") not seen this interval");
  } else {
    logLine("[DOOR] BLE peer2 (" + String(PEER2_NAME) + ") RSSI = " + String(peer2Rssi));
  }
}

void initSensors() {
  Wire.begin(16, 15);

  if (!bme.begin(0x76)) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    bme_ok = false;
  } else {
    bme_ok = true;
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);  
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150); 
    Serial.println("[DOOR] BME680 initialized");
  }

  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    ads_ok = false;
  } else {
    ads_ok = true;
    Serial.println("[DOOR] ADS1015 initialized");
  }
}

DoorSample readDoorSample() {
  DoorSample s;
  s.temp_c   = NAN;
  s.hum_pct  = NAN;
  s.mic_v    = NAN;
  s.light_v  = NAN;

  if (bme_ok) {
    if (bme.performReading()) {
      float raw_temp_c  = bme.temperature;
      float raw_hum_pct = bme.humidity;

      float temp_cal = applyCal(raw_temp_c,  calTempDoor_a, calTempDoor_b, calTempDoor_enabled);
      float hum_cal  = applyCal(raw_hum_pct, calHumDoor_a,  calHumDoor_b,  calHumDoor_enabled);

      s.temp_c  = kf_temp.update(temp_cal);
      s.hum_pct = kf_hum.update(hum_cal);
    } else {
      Serial.println("[DOOR] BME680 performReading() failed");
    }
  }

  if (ads_ok) {
    int16_t mic_raw_counts   = ads.readADC_SingleEnded(0);
    int16_t light_raw_counts = ads.readADC_SingleEnded(2);

    float mic_v_raw   = ads.computeVolts(mic_raw_counts);
    float light_v_raw = ads.computeVolts(light_raw_counts);

    float mic_v_cal   = applyCal(mic_v_raw,   calMic_a,   calMic_b,   calMic_enabled);
    float light_v_cal = applyCal(light_v_raw, calLight_a, calLight_b, calLight_enabled);

    s.mic_v   = kf_mic.update(mic_v_cal);
    s.light_v = kf_light.update(light_v_cal);
  }

  return s;
}

void sendFeature() {
  unsigned long now_ms = millis();
  if (lastBleScanMillis == 0 || now_ms - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[DOOR] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now_ms;
  }

  DoorSample env = readDoorSample();

  unsigned long ts = millis() / 1000UL;

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\","; 
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{"; 

  json += "\"temp_door_c\":";
  json += isnan(env.temp_c) ? "null" : String(env.temp_c, 2);
  json += ",";

  json += "\"hum_door_pct\":";
  json += isnan(env.hum_pct) ? "null" : String(env.hum_pct, 1);
  json += ",";

  // mic / light in volts (filtered)
  json += "\"mic_v\":";
  json += isnan(env.mic_v) ? "null" : String(env.mic_v, 3);
  json += ",";

  json += "\"light_door_v\":";
  json += isnan(env.light_v) ? "null" : String(env.light_v, 3);
  json += ",";

  // BLE RSSI
  json += "\"ble_peer1_rssi\":";
  json += (lastPeer1Rssi == RSSI_INVALID ? String("null") : String(lastPeer1Rssi));
  json += ",";

  json += "\"ble_peer2_rssi\":";
  json += (lastPeer2Rssi == RSSI_INVALID ? "null" : String(lastPeer2Rssi));

  json += "}}";
  json += "\n";

  client.print(json);
  logLine("[DOOR] Sent feature: " + json);
}

void handleIncomingData() {
  while (client.available() > 0) {
    char c = (char)client.read();
    if (c == '\n') {
      if (rxBuffer.length() > 0) {
        logLine("[DOOR] Received plan line: " + rxBuffer);

        int idx = rxBuffer.indexOf("\"state\"");
        if (idx >= 0) {
          int colon = rxBuffer.indexOf(':', idx);
          int q1 = rxBuffer.indexOf('"', colon + 1);
          int q2 = rxBuffer.indexOf('"', q1 + 1);
          if (q1 >= 0 && q2 > q1) {
            String state = rxBuffer.substring(q1 + 1, q2);
            logLine("[DOOR] Parsed state = " + state);
          }
        }
        rxBuffer = "";
      }
    } else {
      rxBuffer += c;
      if (rxBuffer.length() > 1024) {
        rxBuffer = "";
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  logLine("[DOOR] Third node (door) starting...");

  initCalibration();  

  initSensors();
  initBLE();
  connectWifi();

  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis  = millis();
  lastBleScanMillis  = 0;  
}

void loop() {
  if (!client.connected()) {
    logLine("[DOOR] Disconnected from server, reconnecting...");
    client.stop();
    while (WiFi.status() != WL_CONNECTED) {
      connectWifi();
    }
    while (!connectToServer()) {
      delay(2000);
    }
    sendHello();
  }

  unsigned long now = millis();

  if (now - lastFeatureMillis >= FEATURE_INTERVAL_MS) {
    sendFeature();
    lastFeatureMillis = now;
  }

  handleIncomingData();
  delay(10);
}

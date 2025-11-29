#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <math.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_ADS1X15.h>          // ADS1015
#include "Adafruit_BME680.h"           // BME680

// BLE
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

// ----------------------------------------------------------------------
// WiFi / server config
// ----------------------------------------------------------------------
const char* WIFI_SSID     = "Mehrali68";
const char* WIFI_PASSWORD = "4165659393";

const char* SERVER_IP     = "10.0.0.32";
const uint16_t SERVER_PORT = 5000;

const char* NODE_ID       = "door";        // JSON node name
const char* NODE_BLE_NAME = "door-node";   // BLE advertised name

// These must match NODE_BLE_NAME used on your other two ESP32s
const char* PEER1_NAME    = "window-node";
const char* PEER2_NAME    = "bed-node";

WiFiClient client;

// how often to send a feature message (ms)
const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
unsigned long lastFeatureMillis  = 0;

String rxBuffer;

// ----------------------------------------------------------------------
// I2C + sensors
// ----------------------------------------------------------------------
// I2C0 on SDA=16, SCL=15 (BME680, ADS1015)
Adafruit_BME680 bme;
Adafruit_ADS1015 ads;

// Flags
bool bme_ok  = false;
bool ads_ok  = false;

// (kept in case you later want multi-sample noise stats)
const int MIC_SAMPLES = 40;

// ----------------------------------------------------------------------
// BLE globals / config
// ----------------------------------------------------------------------
BLEScan*        pBLEScan      = nullptr;
BLEAdvertising* pAdvertising  = nullptr;
const int BLE_SCAN_TIME_SEC   = 2;
const int RSSI_INVALID        = -999;

const unsigned long BLE_SCAN_INTERVAL_MS = 5UL * 60UL * 1000UL;
unsigned long lastBleScanMillis = 0;   // 0 => never scanned yet

int lastPeer1Rssi = RSSI_INVALID;
int lastPeer2Rssi = RSSI_INVALID;

// ----------------------------------------------------------------------
// Sample struct (minimal: temp, hum, mic, light)
// ----------------------------------------------------------------------
struct DoorSample {
  float temp_c;
  float hum_pct;
  float mic_v;
  float light_v;
};

// ----------------------------------------------------------------------
// Simple 1D Kalman filter for scalar signals
// ----------------------------------------------------------------------
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
      // Don't update on invalid measurement, just return previous estimate (or NaN if never init)
      return initialized ? x : NAN;
    }

    if (!initialized) {
      x = z;
      P = 1.0f;
      initialized = true;
      return x;
    }

    // Predict
    P += Q;

    // Update
    float K = P / (P + R);
    x = x + K * (z - x);
    P = (1.0f - K) * P;

    return x;
  }
};

// One Kalman filter per sensor signal
Kalman1D kf_temp(0.01f, 0.25f);   // temperature: low noise
Kalman1D kf_hum(0.01f, 1.0f);     // humidity
Kalman1D kf_mic(0.1f, 10.0f);     // mic voltage (noisy)
Kalman1D kf_light(0.05f, 5.0f);   // light voltage

// ----------------------------------------------------------------------
// Logging helper
// ----------------------------------------------------------------------
void logLine(const String& msg) {
  Serial.println(msg);
}

// ----------------------------------------------------------------------
// WiFi + TCP helpers
// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// BLE init + scan
// ----------------------------------------------------------------------
void initBLE() {
  logLine("[DOOR] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);

  // Advertise this node so others can see us
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[DOOR] BLE advertising started as: " + String(NODE_BLE_NAME));

  // Prepare scanner to look for other ESP32 nodes
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

// ----------------------------------------------------------------------
// Sensor init
// ----------------------------------------------------------------------
void initSensors() {
  // I2C bus (BME, ADC)
  Wire.begin(16, 15);

  // BME680
  if (!bme.begin(0x76)) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    bme_ok = false;
  } else {
    bme_ok = true;
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);  // still configured but unused
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150); // 320°C for 150 ms (unused, but fine)
    Serial.println("[DOOR] BME680 initialized");
  }

  // ADS1015
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    ads_ok = false;
  } else {
    ads_ok = true;
    Serial.println("[DOOR] ADS1015 initialized");
  }
}

// ----------------------------------------------------------------------
// Sample read + Kalman filtering (temp, hum, mic, light only)
// ----------------------------------------------------------------------
DoorSample readDoorSample() {
  DoorSample s;
  s.temp_c   = NAN;
  s.hum_pct  = NAN;
  s.mic_v    = NAN;
  s.light_v  = NAN;

  // BME680
  if (bme_ok) {
    if (bme.performReading()) {
      float raw_temp_c  = bme.temperature;
      float raw_hum_pct = bme.humidity;

      s.temp_c  = kf_temp.update(raw_temp_c);
      s.hum_pct = kf_hum.update(raw_hum_pct);
    } else {
      Serial.println("[DOOR] BME680 performReading() failed");
    }
  }

  // ADS1015: read channels, convert to volts, then filter
  if (ads_ok) {
    int16_t mic_raw   = ads.readADC_SingleEnded(0);
    // Channel 1 unused
    int16_t light_raw = ads.readADC_SingleEnded(2);

    float mic_v   = ads.computeVolts(mic_raw);
    float light_v = ads.computeVolts(light_raw);

    s.mic_v   = kf_mic.update(mic_v);
    s.light_v = kf_light.update(light_v);
  }

  return s;
}

// ----------------------------------------------------------------------
// Feature send: env + BLE → JSON
// ----------------------------------------------------------------------
void sendFeature() {
  // BLE scan (immediate the first time, then every 5 min)
  unsigned long now_ms = millis();
  if (lastBleScanMillis == 0 || now_ms - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[DOOR] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now_ms;
  }

  // Read sensors (already Kalman-filtered)
  DoorSample env = readDoorSample();

  unsigned long ts = millis() / 1000UL;

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // BME (filtered values)
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

// ----------------------------------------------------------------------
// Handle plan messages from brain
// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// setup / loop
// ----------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  logLine("[DOOR] Third node (door) starting...");

  initSensors();
  initBLE();
  connectWifi();

  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis  = millis();
  lastBleScanMillis  = 0;  // force immediate first BLE scan
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

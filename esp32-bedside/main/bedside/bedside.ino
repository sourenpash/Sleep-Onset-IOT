#include <WiFi.h>
#include "DHT.h"
#include <Wire.h>
#include <BH1750.h>

// ====== BLE includes ======
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
// ==========================

// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "Mehrali68";
const char* WIFI_PASSWORD = "4165659393";

const char* SERVER_IP     = "10.0.0.32";
const uint16_t SERVER_PORT = 5000;

// Logical node ID (for JSON) and BLE device name (for advertising)
const char* NODE_ID       = "bedside";
const char* NODE_BLE_NAME = "bed-node";

// Names of the OTHER two ESP32 nodes, as they advertise over BLE.
// Match these to NODE_BLE_NAME of your window + door nodes.
const char* PEER1_NAME    = "window-node";
const char* PEER2_NAME    = "door-node";
// =============================

// DHT22 setup (bedside air)
#define DHTTYPE DHT22
const int DHT_BED_PIN = 33;   // TODO: set to your actual DHT pin

DHT dhtBed(DHT_BED_PIN, DHTTYPE);

// BH1750 light sensor on SDA/SCL
BH1750 lightMeter;

// WiFi / TCP client
WiFiClient client;

// how often to send a feature message (ms)
const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
unsigned long lastFeatureMillis  = 0;

String rxBuffer;

// simple helper to print and flush
void logLine(const String& msg) {
  Serial.println(msg);
}

// ---- Simple 1D Kalman filter for temp/humidity ----
struct Kalman1D {
  float x;          // current estimate
  float P;          // current error covariance
  float Q;          // process noise variance
  float R;          // measurement noise variance
  bool initialized; // have we seen a first measurement yet?

  void init(float q, float r) {
    Q = q;
    R = r;
    x = 0.0f;
    P = 1.0f;       // initial variance guess
    initialized = false;
  }

  float update(float z) {
    // If NaN, skip update and just return current estimate (or NaN if uninitialized)
    if (isnan(z)) {
      return initialized ? x : NAN;
    }

    if (!initialized) {
      // First measurement sets initial state
      x = z;
      P = 1.0f;
      initialized = true;
      return x;
    }

    // Prediction step (assume state roughly constant between samples)
    P = P + Q;

    // Update step
    float K = P / (P + R); // Kalman gain
    x = x + K * (z - x);   // new estimate
    P = (1.0f - K) * P;    // new covariance

    return x;
  }
};

Kalman1D kfTempBed;
Kalman1D kfHumBed;

// ---- BLE globals ----
BLEScan*        pBLEScan      = nullptr;
BLEAdvertising* pAdvertising  = nullptr;
const int BLE_SCAN_TIME_SEC   = 2;     // scan duration when we do scan
const int RSSI_INVALID        = -999;  // sentinel for "not seen"

// BLE scan cadence: only triangulate every 5 minutes
const unsigned long BLE_SCAN_INTERVAL_MS = 5UL * 60UL * 1000UL;
unsigned long lastBleScanMillis = 0;

// cached last known RSSI values from last triangulation
int lastPeer1Rssi = RSSI_INVALID;
int lastPeer2Rssi = RSSI_INVALID;
// --------------------------------

// ---- WiFi + TCP connect helpers ----

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  logLine("[BED] Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  logLine("[BED] WiFi connected");
  Serial.print("[BED] IP: ");
  Serial.println(WiFi.localIP());
}

bool connectToServer() {
  logLine("[BED] Connecting to server...");
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    logLine("[BED] Failed to connect to server");
    return false;
  }
  logLine("[BED] Connected to server");
  return true;
}

void sendHello() {
  String hello = "{\"type\":\"hello\",\"node\":\"";
  hello += NODE_ID;
  hello += "\"}\n";
  client.print(hello);
  logLine("[BED] Sent hello: " + hello);
}

// ---- DHT22 helpers ----

float safeReadTemp(DHT& dht, const char* label) {
  float t = dht.readTemperature(); // °C
  if (isnan(t)) {
    logLine(String("[BED] Failed to read temperature from ") + label);
    return NAN;
  }
  return t;
}

float safeReadHum(DHT& dht, const char* label) {
  float h = dht.readHumidity(); // %
  if (isnan(h)) {
    logLine(String("[BED] Failed to read humidity from ") + label);
    return NAN;
  }
  return h;
}

// ---- BH1750 helper ----

float safeReadLux(BH1750& sensor, const char* label) {
  float lux = sensor.readLightLevel();  // lux
  if (lux < 0.0f || lux > 120000.0f) {
    logLine(String("[BED] Failed to read lux from ") + label);
    return NAN;
  }
  return lux;
}

// ---- BLE init ----
void initBLE() {
  logLine("[BED] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);

  // Advertise this node so other ESP32s can see it
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[BED] BLE advertising started as: " + String(NODE_BLE_NAME));

  // Prepare scanner to look for other ESP32 nodes
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);  // active scan gives RSSI + name
  pBLEScan->setInterval(100);     // ms
  pBLEScan->setWindow(80);        // ms (must be <= interval)
  logLine("[BED] BLE scanner ready");
}

// ---- BLE scan helper to find other ESP32 nodes ----
void scanForPeers(int& peer1Rssi, int& peer2Rssi) {
  peer1Rssi = RSSI_INVALID;
  peer2Rssi = RSSI_INVALID;

  if (!pBLEScan) return;

  // In this BLE lib, start() returns a pointer to BLEScanResults
  BLEScanResults* results = pBLEScan->start(BLE_SCAN_TIME_SEC, false);
  if (!results) {
    pBLEScan->clearResults();
    logLine("[BED] BLE scan returned null results");
    return;
  }

  int count = results->getCount();

  for (int i = 0; i < count; ++i) {
    BLEAdvertisedDevice dev = results->getDevice(i);

    // Normalize name into Arduino String
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
    logLine("[BED] BLE peer1 (" + String(PEER1_NAME) + ") not seen this interval");
  } else {
    logLine("[BED] BLE peer1 (" + String(PEER1_NAME) + ") RSSI = " + String(peer1Rssi));
  }

  if (peer2Rssi == RSSI_INVALID) {
    logLine("[BED] BLE peer2 (" + String(PEER2_NAME) + ") not seen this interval");
  } else {
    logLine("[BED] BLE peer2 (" + String(PEER2_NAME) + ") RSSI = " + String(peer2Rssi));
  }
}

// ---- Feature generation from DHT22 + Kalman + BH1750 + BLE ----

void sendFeature() {
  // Raw readings from bedside DHT
  float temp_bed_raw  = safeReadTemp(dhtBed, "bedside DHT22");
  float hum_bed_raw   = safeReadHum(dhtBed, "bedside DHT22");

  // Kalman-filtered values
  float temp_bed_filt = kfTempBed.update(temp_bed_raw);
  float hum_bed_filt  = kfHumBed.update(hum_bed_raw);

  // Light level from BH1750 (lux)
  float lux_bed       = safeReadLux(lightMeter, "BH1750");

  logLine("[BED] T_bed=" + String(temp_bed_filt, 2) +
          "C, RH_bed=" + String(hum_bed_filt, 2) +
          "%, Lux=" + String(lux_bed, 1));

  // ---- BLE scan for triangulation every 5 minutes ----
  unsigned long now = millis();
  if (now - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[BED] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now;
  }

  // Build JSON message
  unsigned long ts = (unsigned long)(millis() / 1000UL); // pseudo-timestamp

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered bedside values
  json += "\"temp_bed_c\":";
  json += isnan(temp_bed_filt) ? "null" : String(temp_bed_filt, 2);
  json += ",";

  json += "\"hum_bed_pct\":";
  json += isnan(hum_bed_filt) ? "null" : String(hum_bed_filt, 2);
  json += ",";

  // Light (lux)
  json += "\"lux_bed\":";
  json += isnan(lux_bed) ? "null" : String(lux_bed, 1);
  json += ",";

  // BLE RSSI fields (cached, updated every 5 minutes)
  json += "\"ble_peer1_rssi\":";
  json += (lastPeer1Rssi == RSSI_INVALID ? "null" : String(lastPeer1Rssi));
  json += ",";

  json += "\"ble_peer2_rssi\":";
  json += (lastPeer2Rssi == RSSI_INVALID ? "null" : String(lastPeer2Rssi));

  json += "}}";
  json += "\n";

  client.print(json);
  logLine("[BED] Sent feature: " + json);
}

// ---- Receive plan (newline-delimited JSON) ----
void handleIncomingData() {
  while (client.available() > 0) {
    char c = (char)client.read();
    if (c == '\n') {
      if (rxBuffer.length() > 0) {
        // We have a complete line of JSON
        logLine("[BED] Received plan line: " + rxBuffer);

        // Very dumb parse: look for "state":"..."
        int idx = rxBuffer.indexOf("\"state\"");
        if (idx >= 0) {
          int colon = rxBuffer.indexOf(':', idx);
          int quote1 = rxBuffer.indexOf('"', colon + 1);
          int quote2 = rxBuffer.indexOf('"', quote1 + 1);
          if (quote1 >= 0 && quote2 > quote1) {
            String state = rxBuffer.substring(quote1 + 1, quote2);
            logLine("[BED] Parsed state = " + state);
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

// ---- Setup & loop ----

void setup() {
  Serial.begin(115200);
  delay(1000);
  logLine("[BED] Bedside node starting...");

  // Init DHT sensor
  dhtBed.begin();
  logLine("[BED] DHT22 initialized");

  // Init Kalman filters
  kfTempBed.init(0.01f, 0.5f);  // temp: slow change, moderate noise
  kfHumBed.init( 0.05f, 2.0f);  // humidity: a bit noisier

  // Init I2C + BH1750
  Wire.begin();   // SDA/SCL default pins (GPIO 21/22 on many ESP32 dev boards)
  if (lightMeter.begin()) {
    logLine("[BED] BH1750 initialized");
  } else {
    logLine("[BED] BH1750 init FAILED");
  }

  // Init BLE (advertise + scanner)
  initBLE();

  // WiFi + TCP
  connectWifi();

  // Try to connect to server; keep retrying
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis  = millis();
  lastBleScanMillis  = millis();  // first BLE scan after 5 minutes
}

void loop() {
  // If connection dropped, try to reconnect
  if (!client.connected()) {
    logLine("[BED] Disconnected from server, reconnecting...");
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

  // Periodically send feature
  if (now - lastFeatureMillis >= FEATURE_INTERVAL_MS) {
    sendFeature();
    lastFeatureMillis = now;
  }

  // Handle any incoming sleep plan JSON
  handleIncomingData();

  delay(10);
}

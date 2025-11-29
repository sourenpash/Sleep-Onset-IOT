#include <WiFi.h>
#include "DHT.h"

// ====== BLE includes ======
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
// ==========================

// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "Mehrali68";
const char* WIFI_PASSWORD = "4165659393";

const char* SERVER_IP     = "10.0.0.31";
const uint16_t SERVER_PORT = 5000;

// Logical node ID (for JSON) and BLE device name (for advertising)
const char* NODE_ID       = "window";
const char* NODE_BLE_NAME = "window-node";

// Names of the OTHER two ESP32 nodes, as they advertise over BLE.
// Match these to NODE_BLE_NAME of your bedside + door nodes.
const char* PEER1_NAME    = "door-node";
const char* PEER2_NAME    = "bed-node";
// =============================

// DHT22 setup
#define DHTTYPE DHT22
const int DHT_INSIDE_PIN = 25;   // inside sensor
const int DHT_OUTSIDE_PIN = 26;  // outside sensor

DHT dhtInside(DHT_INSIDE_PIN, DHTTYPE);
DHT dhtOutside(DHT_OUTSIDE_PIN, DHTTYPE);

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

Kalman1D kfTempInside;
Kalman1D kfHumInside;

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
  logLine("[WIN] Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  logLine("[WIN] WiFi connected");
  Serial.print("[WIN] IP: ");
  Serial.println(WiFi.localIP());
}

bool connectToServer() {
  logLine("[WIN] Connecting to server...");
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    logLine("[WIN] Failed to connect to server");
    return false;
  }
  logLine("[WIN] Connected to server");
  return true;
}

void sendHello() {
  String hello = "{\"type\":\"hello\",\"node\":\"";
  hello += NODE_ID;
  hello += "\"}\n";
  client.print(hello);
  logLine("[WIN] Sent hello: " + hello);
}

// ---- DHT22 helpers ----

float safeReadTemp(DHT& dht, const char* label) {
  float t = dht.readTemperature(); // °C
  if (isnan(t)) {
    logLine(String("[WIN] Failed to read temperature from ") + label);
    return NAN;
  }
  return t;
}

float safeReadHum(DHT& dht, const char* label) {
  float h = dht.readHumidity(); // %
  if (isnan(h)) {
    logLine(String("[WIN] Failed to read humidity from ") + label);
    return NAN;
  }
  return h;
}

// ---- BLE init ----
void initBLE() {
  logLine("[WIN] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);

  // Advertise this node so other ESP32s can see it
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[WIN] BLE advertising started as: " + String(NODE_BLE_NAME));

  // Prepare scanner to look for other ESP32 nodes
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);  // active scan gives RSSI + name
  pBLEScan->setInterval(100);     // ms
  pBLEScan->setWindow(80);        // ms (must be <= interval)
  logLine("[WIN] BLE scanner ready");
}

// ---- BLE scan helper to find other ESP32 nodes ----
void scanForPeers(int& peer1Rssi, int& peer2Rssi) {
  peer1Rssi = RSSI_INVALID;
  peer2Rssi = RSSI_INVALID;

  if (!pBLEScan) return;

  // In your BLE lib, start() returns a pointer to BLEScanResults
  BLEScanResults* results = pBLEScan->start(BLE_SCAN_TIME_SEC, false);
  if (!results) {
    pBLEScan->clearResults();
    logLine("[WIN] BLE scan returned null results");
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
    logLine("[WIN] BLE peer1 (" + String(PEER1_NAME) + ") not seen this interval");
  } else {
    logLine("[WIN] BLE peer1 (" + String(PEER1_NAME) + ") RSSI = " + String(peer1Rssi));
  }

  if (peer2Rssi == RSSI_INVALID) {
    logLine("[WIN] BLE peer2 (" + String(PEER2_NAME) + ") not seen this interval");
  } else {
    logLine("[WIN] BLE peer2 (" + String(PEER2_NAME) + ") RSSI = " + String(peer2Rssi));
  }
}

// ---- Feature generation from DHT22s + Kalman + BLE ----

void sendFeature() {
  // Raw readings from inside DHT
  float temp_win_raw  = safeReadTemp(dhtInside, "inside DHT22");
  float hum_win_raw   = safeReadHum(dhtInside, "inside DHT22");

  // Raw readings from outside DHT
  float temp_out_c    = safeReadTemp(dhtOutside, "outside DHT22");
  float hum_out_pct   = safeReadHum(dhtOutside, "outside DHT22");

  // Kalman-filtered inside temp & humidity
  float temp_win_filt = kfTempInside.update(temp_win_raw);
  float hum_win_filt  = kfHumInside.update(hum_win_raw);

  logLine("[WIN] T_in=" + String(temp_win_filt, 2) +
          "C, T_out=" + String(temp_out_c, 2) +
          "C, RH_in=" + String(hum_win_filt, 2) +
          "%, RH_out=" + String(hum_out_pct, 2) + "%");

  // ---- BLE scan for triangulation every 5 minutes ----
  unsigned long now = millis();
  if (now - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[WIN] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now;
  }

  // ---- Build JSON message ----
  unsigned long ts = (unsigned long)(millis() / 1000UL); // pseudo-timestamp

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered inside values (main fields)
  json += "\"temp_win_c\":";
  json += isnan(temp_win_filt) ? "null" : String(temp_win_filt, 2);
  json += ",";

  json += "\"hum_win_pct\":";
  json += isnan(hum_win_filt) ? "null" : String(hum_win_filt, 2);
  json += ",";

  // Outside values (raw)
  json += "\"temp_out_c\":";
  json += isnan(temp_out_c) ? "null" : String(temp_out_c, 2);
  json += ",";

  json += "\"hum_out_pct\":";
  json += isnan(hum_out_pct) ? "null" : String(hum_out_pct, 2);
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
  logLine("[WIN] Sent feature: " + json);
}

// ---- Receive plan (newline-delimited JSON) ----
void handleIncomingData() {
  while (client.available() > 0) {
    char c = (char)client.read();
    if (c == '\n') {
      if (rxBuffer.length() > 0) {
        // We have a complete line of JSON
        logLine("[WIN] Received plan line: " + rxBuffer);

        // Very dumb parse: look for "state":"..."
        int idx = rxBuffer.indexOf("\"state\"");
        if (idx >= 0) {
          int colon = rxBuffer.indexOf(':', idx);
          int quote1 = rxBuffer.indexOf('"', colon + 1);
          int quote2 = rxBuffer.indexOf('"', quote1 + 1);
          if (quote1 >= 0 && quote2 > quote1) {
            String state = rxBuffer.substring(quote1 + 1, quote2);
            logLine("[WIN] Parsed state = " + state);
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
  logLine("[WIN] Window node starting...");

  // Init DHT sensors
  dhtInside.begin();
  dhtOutside.begin();
  logLine("[WIN] DHT22 sensors initialized");

  // Init Kalman filters for inside temp & humidity
  kfTempInside.init(0.01f, 0.5f);  // temp: slow change, moderate noise
  kfHumInside.init( 0.05f, 2.0f);  // humidity: a bit noisier

  // Init BLE (advertise + scanner)
  initBLE();

  // WiFi + TCP
  connectWifi();

  // Try to connect to server; keep retrying
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis = millis();
  lastBleScanMillis = millis();  // so first BLE scan happens after 5 min
}

void loop() {
  // If connection dropped, try to reconnect
  if (!client.connected()) {
    logLine("[WIN] Disconnected from server, reconnecting...");
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

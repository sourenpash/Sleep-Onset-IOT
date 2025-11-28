#include <WiFi.h>
#include "DHT.h"

// BLE includes
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLEAdvertising.h>

// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "Mehrali68";
const char* WIFI_PASSWORD = "4165659393";

// PC / Raspberry Pi brain server IP + port
const char* SERVER_IP     = "10.0.0.43";
const uint16_t SERVER_PORT = 5000;
// =============================

// DHT22 setup (single sensor)
#define DHTTYPE DHT22
const int DHT_INSIDE_PIN = 25;   // single temp/humidity sensor

DHT dhtInside(DHT_INSIDE_PIN, DHTTYPE);

WiFiClient client;

// how often to send a feature message (ms)
const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
const float         DT_SEC              = FEATURE_INTERVAL_MS / 1000.0f;

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
  // This node is the bedside node
  String hello = "{\"type\":\"hello\",\"node\":\"bedside\"}\n";
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

// ---- Feature generation from single DHT + Kalman ----

void sendFeature() {
  // Raw readings from single DHT (at bedside)
  float temp_raw  = safeReadTemp(dhtInside, "DHT22");
  float hum_raw   = safeReadHum(dhtInside, "DHT22");

  // Kalman-filtered temp & humidity
  float temp_filt = kfTempInside.update(temp_raw);
  float hum_filt  = kfHumInside.update(hum_raw);

  logLine("[BED] T_raw=" + String(temp_raw, 2) +
          " T_filt=" + String(temp_filt, 2) +
          " H_raw=" + String(hum_raw, 2) +
          " H_filt=" + String(hum_filt, 2));

  // ---- Build JSON message ----
  unsigned long ts = (unsigned long)(millis() / 1000UL); // pseudo-timestamp

  String json = "{";
  json += "\"node\":\"bedside\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered bedside values (note: names match env_map expectations)
  json += "\"temp_bed_c\":";
  json += isnan(temp_filt) ? "null" : String(temp_filt, 2);
  json += ",";

  json += "\"hum_bed_pct\":";
  json += isnan(hum_filt) ? "null" : String(hum_filt, 2);
  json += ",";

  // No light sensor yet -> publish null for light_bed_lux
  json += "\"light_bed_lux\":null";

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
  logLine("[BED] Bedside node starting (single DHT + BLE)...");

  // Init DHT sensor
  dhtInside.begin();
  logLine("[BED] DHT22 sensor initialized");

  // Init Kalman filters for temp & humidity
  kfTempInside.init(0.01f, 0.5f);  // temp: slow change, moderate noise
  kfHumInside.init( 0.05f, 2.0f);  // humidity: a bit noisier

  // --- BLE advertising: identify this node as NODE_BEDSIDE ---
  BLEDevice::init("NODE_BEDSIDE");   // advertised BLE name
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // optional tuning hints
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  logLine("[BED] BLE advertising as NODE_BEDSIDE");

  // WiFi + TCP
  connectWifi();

  // Try to connect to server; keep retrying
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis = millis();
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

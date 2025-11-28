#include <WiFi.h>
#include "DHT.h"
#include <Preferences.h>

// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "Mehrali68";
const char* WIFI_PASSWORD = "4165659393";

// Raspberry Pi (or dev machine) IP + port
const char* SERVER_IP     = "10.0.0.43";
const uint16_t SERVER_PORT = 5000;
// =============================

// DHT22 setup
#define DHTTYPE DHT22
const int DHT_INSIDE_PIN = 25;   // inside sensor
const int DHT_OUTSIDE_PIN = 26;  // outside sensor

DHT dhtInside(DHT_INSIDE_PIN, DHTTYPE);
DHT dhtOutside(DHT_OUTSIDE_PIN, DHTTYPE);

WiFiClient client;
Preferences prefs;

// how often to send a feature message (ms)
const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
const float         DT_SEC              = FEATURE_INTERVAL_MS / 1000.0f;

unsigned long lastFeatureMillis  = 0;
unsigned long lastPersistMillis  = 0;
const unsigned long PERSIST_INTERVAL_MS = 60UL * 1000UL; // write to flash at most once per minute

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

// ---- Simple first-order cooling model ----

enum WindowMode {
  MODE_CLOSED = 0,
  MODE_OPEN   = 1,
  MODE_OPEN_FAN = 2
};

struct CoolingModel {
  float alpha_closed;
  float alpha_open;
  float alpha_open_fan;
};

CoolingModel model;

WindowMode currentMode = MODE_CLOSED;
WindowMode prevMode    = MODE_CLOSED;

float T_in_prev  = NAN;
float T_out_prev = NAN;

const float ETA_ALPHA   = 0.05f;  // learning rate
const float EPS_DENOM   = 0.2f;   // avoid tiny denominator
const float ALPHA_MIN   = 0.0f;
const float ALPHA_MAX   = 0.5f;   // safety cap

void loadModelFromFlash() {
  prefs.begin("window_model", false);
  model.alpha_closed   = prefs.getFloat("a_closed",   0.001f);
  model.alpha_open     = prefs.getFloat("a_open",     0.01f);
  model.alpha_open_fan = prefs.getFloat("a_open_fan", 0.02f);

  logLine("[WIN] Loaded model alphas:");
  logLine("  alpha_closed   = " + String(model.alpha_closed, 6));
  logLine("  alpha_open     = " + String(model.alpha_open, 6));
  logLine("  alpha_open_fan = " + String(model.alpha_open_fan, 6));
}

void persistModelToFlash() {
  prefs.putFloat("a_closed",   model.alpha_closed);
  prefs.putFloat("a_open",     model.alpha_open);
  prefs.putFloat("a_open_fan", model.alpha_open_fan);
  logLine("[WIN] Persisted model alphas to flash");
}

void updateModelFromSample(float T_in, float T_out) {
  if (isnan(T_in_prev) || isnan(T_out_prev)) {
    T_in_prev  = T_in;
    T_out_prev = T_out;
    prevMode   = currentMode;
    return;
  }

  if (currentMode == prevMode) {
    float denom = (T_out_prev - T_in_prev);
    if (fabs(denom) > EPS_DENOM) {
      float num = (T_in - T_in_prev);
      float alpha_sample = num / denom;

      // Clamp sample to reasonable range
      if (alpha_sample > ALPHA_MIN && alpha_sample < ALPHA_MAX) {
        float* alpha_target = nullptr;
        if (currentMode == MODE_CLOSED)    alpha_target = &model.alpha_closed;
        if (currentMode == MODE_OPEN)      alpha_target = &model.alpha_open;
        if (currentMode == MODE_OPEN_FAN)  alpha_target = &model.alpha_open_fan;

        if (alpha_target) {
          *alpha_target = (1.0f - ETA_ALPHA) * (*alpha_target)
                        + ETA_ALPHA * alpha_sample;
        }
      }
    }
  }

  // Update previous state for next step
  T_in_prev  = T_in;
  T_out_prev = T_out;
  prevMode   = currentMode;
}

// Estimate time (in seconds) to reach a target temp given alpha and dt
float estimateTimeToTarget(float T_in0,
                           float T_out,
                           float T_target,
                           float alpha,
                           float dt_sec) {
  if (alpha <= 0.0f || alpha >= 0.5f) {
    return INFINITY;
  }

  float num   = T_target - T_out;
  float denom = T_in0    - T_out;
  if (fabs(denom) < 0.1f) {
    // Already near outside temp or no usable gradient
    return 0.0f;
  }

  float ratio = num / denom;
  // If ratio not between 0 and 1, target is outside the asymptotic range
  if (ratio <= 0.0f || ratio >= 1.0f) {
    return INFINITY;
  }

  float k = logf(ratio) / logf(1.0f - alpha);
  if (k < 0.0f) k = 0.0f;

  return k * dt_sec;
}

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
  String hello = "{\"type\":\"hello\",\"node\":\"window\"}\n";
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

// ---- Feature generation from DHT22s + Kalman + model learning ----

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

  // ---- Update dynamic model from this sample ----
  if (!isnan(temp_win_filt) && !isnan(temp_out_c)) {
    updateModelFromSample(temp_win_filt, temp_out_c);
  }

  // For now, just demonstrate the time-to-target estimate for a fixed target
  float T_target_demo = 22.0f;  // demo target
  float t_closed   = estimateTimeToTarget(temp_win_filt, temp_out_c,
                                          T_target_demo, model.alpha_closed, DT_SEC);
  float t_open     = estimateTimeToTarget(temp_win_filt, temp_out_c,
                                          T_target_demo, model.alpha_open, DT_SEC);
  float t_open_fan = estimateTimeToTarget(temp_win_filt, temp_out_c,
                                          T_target_demo, model.alpha_open_fan, DT_SEC);

  logLine("[WIN] T_in=" + String(temp_win_filt, 2) +
          " T_out=" + String(temp_out_c, 2) +
          " -> t_closed=" + String(t_closed, 1) +
          "s, t_open=" + String(t_open, 1) +
          "s, t_open_fan=" + String(t_open_fan, 1) + "s");

  // ---- Build JSON message ----
  unsigned long ts = (unsigned long)(millis() / 1000UL); // pseudo-timestamp

  String json = "{";
  json += "\"node\":\"window\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered inside values (main fields)
  json += "\"temp_win_c\":";
  json += isnan(temp_win_filt) ? "null" : String(temp_win_filt, 2);
  json += ",";

  json += "\"hum_win_pct\":";
  json += isnan(hum_win_filt) ? "null" : String(hum_win_filt, 2);
  json += ",";

  // Outside values (raw for now)
  json += "\"temp_out_c\":";
  json += isnan(temp_out_c) ? "null" : String(temp_out_c, 2);
  json += ",";

  json += "\"hum_out_pct\":";
  json += isnan(hum_out_pct) ? "null" : String(hum_out_pct, 2);

  // Optionally, we could also publish learned alpha params for logging/debug
  // json += ",\"alpha_closed\":"   + String(model.alpha_closed, 6);
  // json += ",\"alpha_open\":"     + String(model.alpha_open, 6);
  // json += ",\"alpha_open_fan\":" + String(model.alpha_open_fan, 6);

  json += "}}";
  json += "\n";

  client.print(json);
  logLine("[WIN] Sent feature: " + json);
}

// ---- Receive plan (newline-delimited JSON) ----
// (For now, we only parse 'state' as before. Later you can parse temp target & t_sleep_pred.)
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

  // Load dynamic model from flash (self-learned alphas)
  loadModelFromFlash();

  connectWifi();

  // Try to connect to server; keep retrying
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis = millis();
  lastPersistMillis = millis();
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

  // Periodically persist model to flash
  if (now - lastPersistMillis >= PERSIST_INTERVAL_MS) {
    persistModelToFlash();
    lastPersistMillis = now;
  }

  // Handle any incoming sleep plan JSON
  handleIncomingData();

  delay(10);
}

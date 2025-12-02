#include <WiFi.h>
#include "DHT.h"


#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>


// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";

const char* SERVER_IP     = "10.0.0.32";
const uint16_t SERVER_PORT = 5000;


const char* NODE_ID       = "window";
const char* NODE_BLE_NAME = "window-node";


const char* PEER1_NAME    = "door-node";
const char* PEER2_NAME    = "bed-node";



#define DHTTYPE DHT22
const int DHT_INSIDE_PIN = 25;   // inside sensor
const int DHT_OUTSIDE_PIN = 26;  // outside sensor

DHT dhtInside(DHT_INSIDE_PIN, DHTTYPE);
DHT dhtOutside(DHT_OUTSIDE_PIN, DHTTYPE);

WiFiClient client;

const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
unsigned long lastFeatureMillis  = 0;

String rxBuffer;


void logLine(const String& msg) {
  Serial.println(msg);
}


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
    P = 1.0f;       
    initialized = false;
  }

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

    P = P + Q;

    float K = P / (P + R); 
    x = x + K * (z - x);   
    P = (1.0f - K) * P;    

    return x;
  }
};

Kalman1D kfTempInside;
Kalman1D kfHumInside;



// Inside temperature (°C) at window (DHT inside)
float CAL_TEMP_IN_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };  
float CAL_TEMP_IN_RAW[3]    = { 0.0f, 0.0f, 0.0f };  

// Inside humidity (%) at window
float CAL_HUM_IN_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_HUM_IN_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Outside temperature (°C) at window (DHT outside)
float CAL_TEMP_OUT_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_TEMP_OUT_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Outside humidity (%) at window
float CAL_HUM_OUT_ACTUAL[3] = { 0.0f, 0.0f, 0.0f };
float CAL_HUM_OUT_RAW[3]    = { 0.0f, 0.0f, 0.0f };

// Calibration parameters per sensor: actual ≈ a * raw + b
float calTempIn_a = 1.0f, calTempIn_b = 0.0f; bool calTempIn_enabled = false;
float calHumIn_a  = 1.0f, calHumIn_b  = 0.0f; bool calHumIn_enabled  = false;
float calTempOut_a = 1.0f, calTempOut_b = 0.0f; bool calTempOut_enabled = false;
float calHumOut_a  = 1.0f, calHumOut_b  = 0.0f; bool calHumOut_enabled  = false;


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
  computeCalGeneric(CAL_TEMP_IN_ACTUAL,  CAL_TEMP_IN_RAW,  3,
                    calTempIn_a, calTempIn_b, calTempIn_enabled);
  computeCalGeneric(CAL_HUM_IN_ACTUAL,   CAL_HUM_IN_RAW,   3,
                    calHumIn_a, calHumIn_b, calHumIn_enabled);
  computeCalGeneric(CAL_TEMP_OUT_ACTUAL, CAL_TEMP_OUT_RAW, 3,
                    calTempOut_a, calTempOut_b, calTempOut_enabled);
  computeCalGeneric(CAL_HUM_OUT_ACTUAL,  CAL_HUM_OUT_RAW,  3,
                    calHumOut_a, calHumOut_b, calHumOut_enabled);

  Serial.println("[WIN] Calibration init:");
  Serial.print("  T_in:  ");
  Serial.print(calTempIn_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calTempIn_a, 4);
  Serial.print(" b="); Serial.println(calTempIn_b, 4);

  Serial.print("  H_in:  ");
  Serial.print(calHumIn_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calHumIn_a, 4);
  Serial.print(" b="); Serial.println(calHumIn_b, 4);

  Serial.print("  T_out: ");
  Serial.print(calTempOut_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calTempOut_a, 4);
  Serial.print(" b="); Serial.println(calTempOut_b, 4);

  Serial.print("  H_out: ");
  Serial.print(calHumOut_enabled ? "ON " : "OFF");
  Serial.print(" a="); Serial.print(calHumOut_a, 4);
  Serial.print(" b="); Serial.println(calHumOut_b, 4);
}


BLEScan*        pBLEScan      = nullptr;
BLEAdvertising* pAdvertising  = nullptr;
const int BLE_SCAN_TIME_SEC   = 2;     
const int RSSI_INVALID        = -999;  


const unsigned long BLE_SCAN_INTERVAL_MS = 5UL * 60UL * 1000UL;
unsigned long lastBleScanMillis = 0;


int lastPeer1Rssi = RSSI_INVALID;
int lastPeer2Rssi = RSSI_INVALID;
// --------------------------------


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


void initBLE() {
  logLine("[WIN] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);


  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[WIN] BLE advertising started as: " + String(NODE_BLE_NAME));


  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);  
  pBLEScan->setInterval(100);     // ms
  pBLEScan->setWindow(80);        // ms 
  logLine("[WIN] BLE scanner ready");
}


void scanForPeers(int& peer1Rssi, int& peer2Rssi) {
  peer1Rssi = RSSI_INVALID;
  peer2Rssi = RSSI_INVALID;

  if (!pBLEScan) return;

  BLEScanResults* results = pBLEScan->start(BLE_SCAN_TIME_SEC, false);
  if (!results) {
    pBLEScan->clearResults();
    logLine("[WIN] BLE scan returned null results");
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
    logLine("[WIN] BLE peer1 (" + String(PEER1_NAME) + ") not seen this interval");
  } else {
    logLine("[WIN] BLE peer1 (" + String(PEER1_NAME) + ") RSSI = " + String(peer1Rssi));
  }

  if (peer2Rssi == RSSI_INVALID) {
    logLine("[WIN] BLE peer2 (" + String(PEER2_NAME) + ") RSSI = " + String(peer2Rssi));
  } else {
    logLine("[WIN] BLE peer2 (" + String(PEER2_NAME) + ") RSSI = " + String(peer2Rssi));
  }
}


void sendFeature() {
  // Raw readings from inside DHT
  float temp_win_raw  = safeReadTemp(dhtInside, "inside DHT22");
  float hum_win_raw   = safeReadHum(dhtInside, "inside DHT22");

  // Raw readings from outside DHT
  float temp_out_raw  = safeReadTemp(dhtOutside, "outside DHT22");
  float hum_out_raw   = safeReadHum(dhtOutside, "outside DHT22");

  // Apply calibration
  float temp_win_cal = applyCal(temp_win_raw, calTempIn_a,  calTempIn_b,  calTempIn_enabled);
  float hum_win_cal  = applyCal(hum_win_raw,  calHumIn_a,   calHumIn_b,   calHumIn_enabled);
  float temp_out_cal = applyCal(temp_out_raw, calTempOut_a, calTempOut_b, calTempOut_enabled);
  float hum_out_cal  = applyCal(hum_out_raw,  calHumOut_a,  calHumOut_b,  calHumOut_enabled);

  // Kalman-filtered inside temp & humidity
  float temp_win_filt = kfTempInside.update(temp_win_cal);
  float hum_win_filt  = kfHumInside.update(hum_win_cal);

  logLine("[WIN] T_in=" + String(temp_win_filt, 2) +
          "C, T_out=" + String(temp_out_cal, 2) +
          "C, RH_in=" + String(hum_win_filt, 2) +
          "%, RH_out=" + String(hum_out_cal, 2) + "%");

 
  unsigned long now = millis();
  if (now - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[WIN] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now;
  }

 
  unsigned long ts = (unsigned long)(millis() / 1000UL); 

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\",";  
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered inside values (main fields used by model/env_map)
  json += "\"temp_win_c\":";
  json += isnan(temp_win_filt) ? "null" : String(temp_win_filt, 2);
  json += ",";

  json += "\"hum_win_pct\":";
  json += isnan(hum_win_filt) ? "null" : String(hum_win_filt, 2);
  json += ",";

  // Explicit aliases for inside (for dashboard / future models)
  json += "\"temp_win_in_c\":";
  json += isnan(temp_win_filt) ? "null" : String(temp_win_filt, 2);
  json += ",";

  json += "\"hum_win_in_pct\":";
  json += isnan(hum_win_filt) ? "null" : String(hum_win_filt, 2);
  json += ",";

  // Outside values (calibrated)
  json += "\"temp_out_c\":";
  json += isnan(temp_out_cal) ? "null" : String(temp_out_cal, 2);
  json += ",";

  json += "\"hum_out_pct\":";
  json += isnan(hum_out_cal) ? "null" : String(hum_out_cal, 2);
  json += ",";

  // Explicit aliases for outside (for dashboard)
  json += "\"temp_win_out_c\":";
  json += isnan(temp_out_cal) ? "null" : String(temp_out_cal, 2);
  json += ",";

  json += "\"hum_win_out_pct\":";
  json += isnan(hum_out_cal) ? "null" : String(hum_out_cal, 2);
  json += ",";


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


void handleIncomingData() {
  while (client.available() > 0) {
    char c = (char)client.read();
    if (c == '\n') {
      if (rxBuffer.length() > 0) {
       
        logLine("[WIN] Received plan line: " + rxBuffer);

        
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



void setup() {
  Serial.begin(115200);
  delay(1000);
  logLine("[WIN] Window node starting...");

  
  dhtInside.begin();
  dhtOutside.begin();
  logLine("[WIN] DHT22 sensors initialized");

  
  kfTempInside.init(0.01f, 0.5f);  
  kfHumInside.init( 0.05f, 2.0f);  

  
  initCalibration();


  initBLE();


  connectWifi();

 
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis = millis();
  lastBleScanMillis = millis();  
}

void loop() {
 
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

 
  if (now - lastFeatureMillis >= FEATURE_INTERVAL_MS) {
    sendFeature();
    lastFeatureMillis = now;
  }

 
  handleIncomingData();

  delay(10);
}

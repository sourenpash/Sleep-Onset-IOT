#include <WiFi.h>
#include "DHT.h"
#include <Wire.h>
#include <BH1750.h>
#include <math.h>  


#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>


// ====== CONFIGURE THESE ======
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";

const char* SERVER_IP     = "10.0.0.32"; // SERVER/PC IP ADDRESS
const uint16_t SERVER_PORT = 5000;

// BLE device name (for advertising)
const char* NODE_ID       = "bedside";
const char* NODE_BLE_NAME = "bed-node";


const char* PEER1_NAME    = "window-node";
const char* PEER2_NAME    = "door-node";
// =============================

// DHT22 setup (bedside air)
#define DHTTYPE DHT22
const int DHT_BED_PIN = 33;   

DHT dhtBed(DHT_BED_PIN, DHTTYPE);


BH1750 lightMeter;

WiFiClient client;

// how often to send a feature message (ms)
const unsigned long FEATURE_INTERVAL_MS = 10UL * 1000UL;
unsigned long lastFeatureMillis  = 0;

String rxBuffer;


void logLine(const String& msg) {
  Serial.println(msg);
}

// ---------------------------------------------------------------------------
//  1D Kalman filter for temp/humidity
// ---------------------------------------------------------------------------
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

    // Update step
    float K = P / (P + R); // Kalman gain
    x = x + K * (z - x);   // new estimate
    P = (1.0f - K) * P;    // new covariance

    return x;
  }
};

Kalman1D kfTempBed;
Kalman1D kfHumBed;


// --- Temperature at bedside (°C) ---
const int N_CAL_TEMP_BED = 3;
// actual (trusted) temperature values
const float CAL_TEMP_BED_ACTUAL[N_CAL_TEMP_BED] = {
  23.0f, 23.0f, 23.0f   // EDIT ME: (actual1, actual2, actual3)
};
// raw readings from this node that correspond to those actual values
const float CAL_TEMP_BED_RAW[N_CAL_TEMP_BED] = {
  23.0f, 23.0f, 23.0f   // EDIT ME: (raw1, raw2, raw3)
};

// --- Humidity at bedside (%) ---
const int N_CAL_HUM_BED = 3;
const float CAL_HUM_BED_ACTUAL[N_CAL_HUM_BED] = {
  45.0f, 45.0f, 45.0f   // EDIT ME
};
const float CAL_HUM_BED_RAW[N_CAL_HUM_BED] = {
  45.0f, 45.0f, 45.0f   // EDIT ME
};

// --- Light at bedside (lux) ---
const int N_CAL_LUX_BED = 3;
const float CAL_LUX_BED_ACTUAL[N_CAL_LUX_BED] = {
  100.0f, 100.0f, 100.0f   // EDIT ME
};
const float CAL_LUX_BED_RAW[N_CAL_LUX_BED] = {
  100.0f, 100.0f, 100.0f   // EDIT ME
};

// Calibration coefficients: actual = a * raw + b
float calTempBed_a = 1.0f, calTempBed_b = 0.0f;
bool  calTempBed_enabled = false;

float calHumBed_a  = 1.0f, calHumBed_b  = 0.0f;
bool  calHumBed_enabled  = false;

float calLuxBed_a  = 1.0f, calLuxBed_b  = 0.0f;
bool  calLuxBed_enabled  = false;

void computeCalGeneric(const float actual[], const float raw[], int n,
                       float &a, float &b, bool &enabled) {
  int count = 0;
  float sumRaw = 0.0f, sumAct = 0.0f;
  float sumRaw2 = 0.0f, sumRawAct = 0.0f;

  for (int i = 0; i < n; ++i) {
    float ra = raw[i];
    float ac = actual[i];


    if (ra == 0.0f && ac == 0.0f) {
      continue;
    }

    sumRaw    += ra;
    sumAct    += ac;
    sumRaw2   += ra * ra;
    sumRawAct += ra * ac;
    count++;
  }

  if (count < 2) {

    a = 1.0f;
    b = 0.0f;
    enabled = false;
    return;
  }

  float n_f    = (float)count;
  float denom  = n_f * sumRaw2 - (sumRaw * sumRaw);
  if (denom == 0.0f) {
    a = 1.0f;
    b = 0.0f;
    enabled = false;
    return;
  }

  a = (n_f * sumRawAct - sumRaw * sumAct) / denom;
  b = (sumAct - a * sumRaw) / n_f;
  enabled = true;
}

float applyCal(float raw, float a, float b, bool enabled) {
  if (!enabled) return raw;
  if (isnan(raw)) return raw;
  return a * raw + b;
}


BLEScan*        pBLEScan      = nullptr;
BLEAdvertising* pAdvertising  = nullptr;
const int BLE_SCAN_TIME_SEC   = 2;     
const int RSSI_INVALID        = -999;  

// BLE scan cadence: only triangulate every 5 minutes
const unsigned long BLE_SCAN_INTERVAL_MS = 5UL * 60UL * 1000UL;
unsigned long lastBleScanMillis = 0;


int lastPeer1Rssi = RSSI_INVALID;
int lastPeer2Rssi = RSSI_INVALID;
// --------------------------------


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


float safeReadLux(BH1750& sensor, const char* label) {
  float lux = sensor.readLightLevel();  // lux
  if (lux < 0.0f || lux > 120000.0f) {
    logLine(String("[BED] Failed to read lux from ") + label);
    return NAN;
  }
  return lux;
}


void initBLE() {
  logLine("[BED] Initializing BLE...");
  BLEDevice::init(NODE_BLE_NAME);


  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  logLine("[BED] BLE advertising started as: " + String(NODE_BLE_NAME));


  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);  
  pBLEScan->setInterval(100);     // ms
  pBLEScan->setWindow(80);        // ms 
  logLine("[BED] BLE scanner ready");
}


void scanForPeers(int& peer1Rssi, int& peer2Rssi) {
  peer1Rssi = RSSI_INVALID;
  peer2Rssi = RSSI_INVALID;

  if (!pBLEScan) return;


  BLEScanResults* results = pBLEScan->start(BLE_SCAN_TIME_SEC, false);
  if (!results) {
    pBLEScan->clearResults();
    logLine("[BED] BLE scan returned null results");
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


// Feature generation from DHT22 + calibration + Kalman + BH1750 + BLE

void sendFeature() {
  // Raw readings from bedside DHT
  float temp_bed_raw  = safeReadTemp(dhtBed, "bedside DHT22");
  float hum_bed_raw   = safeReadHum(dhtBed, "bedside DHT22");

  // Light level from BH1750 (raw lux)
  float lux_bed_raw   = safeReadLux(lightMeter, "BH1750");

  // Apply calibration (if enabled) BEFORE filtering
  float temp_bed_cal  = applyCal(temp_bed_raw, calTempBed_a, calTempBed_b, calTempBed_enabled);
  float hum_bed_cal   = applyCal(hum_bed_raw,  calHumBed_a,  calHumBed_b,  calHumBed_enabled);
  float lux_bed_cal   = applyCal(lux_bed_raw,  calLuxBed_a,  calLuxBed_b,  calLuxBed_enabled);

  // Kalman-filtered temp & humidity
  float temp_bed_filt = kfTempBed.update(temp_bed_cal);
  float hum_bed_filt  = kfHumBed.update(hum_bed_cal);

  logLine("[BED] T_bed=" + String(temp_bed_filt, 2) +
          "C, RH_bed="  + String(hum_bed_filt, 2) +
          "%, Lux="     + String(lux_bed_cal, 1));

  
  unsigned long now = millis();
  if (now - lastBleScanMillis >= BLE_SCAN_INTERVAL_MS) {
    logLine("[BED] Running BLE scan for peers...");
    scanForPeers(lastPeer1Rssi, lastPeer2Rssi);
    lastBleScanMillis = now;
  }

 
  unsigned long ts = (unsigned long)(millis() / 1000UL); // pseudo-timestamp

  String json = "{";
  json += "\"node\":\"";
  json += NODE_ID;
  json += "\",";
  json += "\"ts\":" + String(ts) + ",";
  json += "\"sensors\":{";

  // Filtered bedside values (these names are used by the brain/sleep model)
  json += "\"temp_bed_c\":";
  json += isnan(temp_bed_filt) ? "null" : String(temp_bed_filt, 2);
  json += ",";

  json += "\"hum_bed_pct\":";
  json += isnan(hum_bed_filt) ? "null" : String(hum_bed_filt, 2);
  json += ",";

  // Light (calibrated lux; no Kalman here, BH1750 is pretty stable)
  json += "\"lux_bed\":";
  json += isnan(lux_bed_cal) ? "null" : String(lux_bed_cal, 1);
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

  // Compute calibration coefficients from the arrays above
  computeCalGeneric(CAL_TEMP_BED_ACTUAL, CAL_TEMP_BED_RAW,
                    N_CAL_TEMP_BED, calTempBed_a, calTempBed_b, calTempBed_enabled);
  computeCalGeneric(CAL_HUM_BED_ACTUAL,  CAL_HUM_BED_RAW,
                    N_CAL_HUM_BED, calHumBed_a, calHumBed_b, calHumBed_enabled);
  computeCalGeneric(CAL_LUX_BED_ACTUAL,  CAL_LUX_BED_RAW,
                    N_CAL_LUX_BED, calLuxBed_a, calLuxBed_b, calLuxBed_enabled);

  logLine("[BED] Temp calibration: enabled=" + String(calTempBed_enabled ? "true" : "false") +
          " a=" + String(calTempBed_a, 4) + " b=" + String(calTempBed_b, 4));
  logLine("[BED] Hum calibration:  enabled=" + String(calHumBed_enabled ? "true" : "false") +
          " a=" + String(calHumBed_a, 4) + " b=" + String(calHumBed_b, 4));
  logLine("[BED] Lux calibration:  enabled=" + String(calLuxBed_enabled ? "true" : "false") +
          " a=" + String(calLuxBed_a, 4) + " b=" + String(calLuxBed_b, 4));

  // Init I2C + BH1750
  Wire.begin();   
  if (lightMeter.begin()) {
    logLine("[BED] BH1750 initialized");
  } else {
    logLine("[BED] BH1750 init FAILED");
  }


  initBLE();

  
  connectWifi();

 
  while (!connectToServer()) {
    delay(2000);
  }

  sendHello();
  lastFeatureMillis  = millis();
  lastBleScanMillis  = millis();  
}

void loop() {
  
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


  if (now - lastFeatureMillis >= FEATURE_INTERVAL_MS) {
    sendFeature();
    lastFeatureMillis = now;
  }


  handleIncomingData();

  delay(10);
}

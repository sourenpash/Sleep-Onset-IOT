#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

// Names we expect the other ESP32 nodes to advertise
// (Make sure your window and bedside nodes use these in BLEDevice::init)
static const char* NAME_WINDOW  = "NODE_WINDOW";
static const char* NAME_BEDSIDE = "NODE_BEDSIDE";
static const char* NAME_DEHUM   = "NODE_DEHUM";   // if you also advertise from this node later

// Global BLE scan object
BLEScan* pBLEScan = nullptr;

// Simple struct to hold latest RSSI values per node
struct NodeRSSI {
  bool  seen;
  int   rssi;
};

NodeRSSI rssiWindow  = {false, 0};
NodeRSSI rssiBedside = {false, 0};
NodeRSSI rssiDehum   = {false, 0};

// Callback for each advertised BLE device we see
class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) override {
    // In your BLE library, getName() returns an Arduino String, not std::string
    String name = advertisedDevice.getName();
    int rssi = advertisedDevice.getRSSI();

    if (!name.length()) {
      return;  // no name, ignore
    }

    if (name.equals(NAME_WINDOW)) {
      rssiWindow.seen = true;
      rssiWindow.rssi = rssi;
    } else if (name.equals(NAME_BEDSIDE)) {
      rssiBedside.seen = true;
      rssiBedside.rssi = rssi;
    } else if (name.equals(NAME_DEHUM)) {
      rssiDehum.seen = true;
      rssiDehum.rssi = rssi;
    }

    // Debug: uncomment to see all named devices
    // Serial.printf("[BLE-TRI] Device name: %s, RSSI: %d\n", name.c_str(), rssi);
  }
};

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("[BLE-TRI] Triangulation scanner starting...");

  // Initialize BLE; this node can have its own name
  BLEDevice::init("NODE_TRIANGULATOR");

  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks(), true);
  pBLEScan->setActiveScan(true);   // active scan to get names quickly
  pBLEScan->setInterval(100);      // ms
  pBLEScan->setWindow(80);         // ms (must be <= interval)

  Serial.println("[BLE-TRI] Scan configured, ready.");
}

void loop() {
  // Run a scan for N seconds, then process the results from callbacks
  const uint32_t scanTimeSec = 3;  // scan window in seconds
  Serial.println("[BLE-TRI] Starting scan...");
  pBLEScan->start(scanTimeSec, false);  // we don't need the return value here
  Serial.println("[BLE-TRI] Scan done.");

  // Print the latest RSSI per node (if seen)
  if (rssiWindow.seen) {
    Serial.printf("[BLE-TRI] %s RSSI: %d dBm\n", NAME_WINDOW, rssiWindow.rssi);
  } else {
    Serial.printf("[BLE-TRI] %s not seen this scan.\n", NAME_WINDOW);
  }

  if (rssiBedside.seen) {
    Serial.printf("[BLE-TRI] %s RSSI: %d dBm\n", NAME_BEDSIDE, rssiBedside.rssi);
  } else {
    Serial.printf("[BLE-TRI] %s not seen this scan.\n", NAME_BEDSIDE);
  }

  if (rssiDehum.seen) {
    Serial.printf("[BLE-TRI] %s RSSI: %d dBm\n", NAME_DEHUM, rssiDehum.rssi);
  } else {
    Serial.printf("[BLE-TRI] %s not seen this scan.\n", NAME_DEHUM);
  }

  Serial.println();

  // Reset "seen" flags for the next scan cycle
  rssiWindow.seen  = false;
  rssiBedside.seen = false;
  rssiDehum.seen   = false;

  // Clear old results to free RAM
  pBLEScan->clearResults();

  // Small delay between scans
  delay(1000);
}

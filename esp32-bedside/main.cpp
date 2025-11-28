#include <Arduino.h>
#include <WiFi.h>

// TODO: set Wi-Fi credentials and Pi server IP before running on hardware.
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
const char* SERVER_IP = "192.168.1.100";  // replace with Pi IP
const uint16_t SERVER_PORT = 5000;

WiFiClient client;
String inboundBuffer;
unsigned long lastSendMs = 0;
const unsigned long SEND_INTERVAL_MS = 3000;

void connectWifi() {
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());
}

bool connectServer() {
  Serial.print("Connecting to server...");
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    Serial.println(" failed");
    return false;
  }
  Serial.println(" connected");
  return true;
}

void sendHello() {
  client.print("{\"type\":\"hello\",\"node\":\"bedside\"}\n");
  Serial.println("Sent hello");
}

String buildFeatureJson(float temp_bed_c, float hum_bed_pct, float light_bed_lux, float noise_level, float motion_index, unsigned long ts) {
  String payload = "{\"node\":\"bedside\",\"ts\":";
  payload += String(ts);
  payload += ",\"sensors\":{";
  payload += "\"temp_bed_c\":" + String(temp_bed_c, 1) + ",";
  payload += "\"hum_bed_pct\":" + String(hum_bed_pct, 1) + ",";
  payload += "\"light_bed_lux\":" + String(light_bed_lux, 1) + ",";
  payload += "\"noise_level\":" + String(noise_level, 2) + ",";
  payload += "\"motion_index\":" + String(motion_index, 2);
  payload += "}}";
  return payload;
}

void handleIncomingLine(const String& line) {
  int statePos = line.indexOf("\"state\"");
  String state = "UNKNOWN";
  if (statePos >= 0) {
    int colon = line.indexOf(':', statePos);
    int quoteStart = line.indexOf('"', colon + 1);
    int quoteEnd = line.indexOf('"', quoteStart + 1);
    if (quoteStart >= 0 && quoteEnd > quoteStart) {
      state = line.substring(quoteStart + 1, quoteEnd);
    }
  }
  Serial.print("Received plan: state=");
  Serial.println(state);

  // Simulate local decision logic
  String bedLamp = state == "ASLEEP" ? "OFF" : "ON";
  float currentHumidity = 45.0;  // placeholder
  String humidifier = currentHumidity < 45.0 ? "ON" : "OFF";
  Serial.print("Desired bed_lamp -> ");
  Serial.println(bedLamp);
  Serial.print("Desired humidifier -> ");
  Serial.println(humidifier);
}

void setup() {
  Serial.begin(115200);
  Serial.println("Bedside node starting (dummy mode)");

  connectWifi();
  if (connectServer()) {
    sendHello();
  }
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }
  if (!client.connected()) {
    connectServer();
    if (client.connected()) {
      sendHello();
    }
  }

  unsigned long now = millis();
  if (now - lastSendMs > SEND_INTERVAL_MS && client.connected()) {
    static int tick = 0;
    float temp_bed_c = 23.5 - 0.1f * tick;
    float hum_bed_pct = 42.0 + 0.2f * tick;
    float light_bed_lux = 150.0 - 5.0f * tick;
    float noise_level = 0.35f;
    float motion_index = tick % 2 == 0 ? 0.8f : 0.3f;

    String payload = buildFeatureJson(temp_bed_c, hum_bed_pct, light_bed_lux, noise_level, motion_index, now / 1000);
    Serial.println("Sending feature payload:");
    Serial.println(payload);
    client.print(payload);
    client.print("\n");

    ++tick;
    lastSendMs = now;
  }

  while (client.connected() && client.available()) {
    char c = client.read();
    if (c == '\n') {
      if (inboundBuffer.length() > 0) {
        handleIncomingLine(inboundBuffer);
        inboundBuffer = "";
      }
    } else {
      inboundBuffer += c;
    }
  }

  delay(10);
}

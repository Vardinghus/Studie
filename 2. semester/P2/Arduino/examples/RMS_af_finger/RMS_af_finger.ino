#define LGT 450

void setup() {
  Serial.begin(9600);
}

float maal[100];
float sum = 0;
float RMS = 0;
int i = 0;

void loop() {
  while (i < LGT) {
    maal[i] = analogRead(A0);
    i++;
  }

  for (int x = 0; x < LGT; x++) {
    sum += maal[x] * maal[x];
  }
  RMS = sqrt(sum / LGT);
  Serial.println(RMS);
  delay(500);
  sum = 0;
  RMS = 0;
  i = 0;
}

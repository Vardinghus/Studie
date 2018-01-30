int groen1 = 5;
int groen2 = 7;
int roed1 = 12;
int roed2 = 8;
int player1 = 13;
int player2 = 2;
int i = 0;
int reaktion1 = 0;
int reaktion2 = 0;
int startknap = 10;

void setup() {
  Serial.begin(9600);
  pinMode(groen1, OUTPUT);
  pinMode(groen2, OUTPUT);
  pinMode(roed1, OUTPUT);
  pinMode(roed2, OUTPUT);
  pinMode(player1, INPUT_PULLUP);
  pinMode(player2, INPUT_PULLUP);
  pinMode(startknap, INPUT_PULLUP);
}

int ar1[5];
int ar2[5];
int sejr[5];

typedef struct Data {
  long Reaktionstid[5];
  long Gennemsnit;
};

Data Spillere[1];

void loop() {

  while (i < 5) {

    if (digitalRead(startknap) == LOW) {

      delay(1000);
      digitalWrite(groen1, HIGH);
      delay(500);
      digitalWrite(groen2, HIGH);
      delay(random(5000, 2000));
      digitalWrite(groen1, LOW);
      digitalWrite(groen2, LOW);

      int start = millis();
      int h1 = 0;
      int h2 = 0;

      while (h1 < 1 || h2 < 1) {
        if (digitalRead(player1) == LOW && h1 < 1) {
          reaktion1 = millis() - start;
          Spillere[0].Reaktionstid[i] = reaktion1;
          h1 += 1;

        }
        if (digitalRead(player2) == LOW && h2 < 1) {
          reaktion2 = millis() - start;
          Spillere[1].Reaktionstid[i] = reaktion2;
          h2 += 1;
        }
      }

      if (reaktion1 < reaktion2) {
        sejr[i] = 1;
        digitalWrite(roed1, HIGH);
      }
      else {
        sejr[i] = 2;
        digitalWrite(roed2, HIGH);
      }

      delay(1000);
      digitalWrite(roed1, LOW);
      digitalWrite(roed2, LOW);
      i++;
    }
  }

  int avg1 = 0;
  int avg2 = 0;

  for (int k = 0; k < 4; k++) {
    avg1 = avg1 + Spillere[0].Reaktionstid[k];
    avg2 = avg2 + Spillere[0].Reaktionstid[k];
    Spillere[0].Gennemsnit = (avg1) / 5;
    Spillere[1].Gennemsnit = (avg2) / 5;

    for (int j = 0; j < 5; j++) {
      Serial.print(j);
      Serial.print(". omgang");
      Serial.print("  ");
      Serial.print(Spillere[0].Reaktionstid[j]);
      Serial.print("  ");
      Serial.print(Spillere[1].Reaktionstid[j]);
      Serial.print("  ");
      Serial.println(sejr[j]);
    }
    Serial.println("Gennemsnit for spiller 1: ");
    Serial.print(avg1);
    Serial.println("Gennemsnit for spiller 2: ");
    Serial.print(avg2);
    delay(100000);
  }
}

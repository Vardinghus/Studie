
int var = 20;
int *p = &var;

void setup() {
  Serial.begin(9600);
}



void loop() {
  Serial.println((int)&var);
  delay(500);
}




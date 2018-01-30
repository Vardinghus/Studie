void setup() {
  Serial.begin(9600)
  dec = 7
}
int a = 10;
int b = 6;


void loop() {
 Serial.println(a/b);
}

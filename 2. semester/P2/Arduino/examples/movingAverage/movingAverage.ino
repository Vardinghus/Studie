
#define k 20
float In[k];
float Out[k];

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < k; i++) {
    In[i] = random(0, 30);
  }
}
int i = 0;
void loop() {
  if (i == 0) {
    Out[i] = In[i] / 3;
  }
  else if (i == 1) {
    Out[i] = (In[i] + In[i - 1]) / 3;
  }
  else {
    Out[i] = (In[i] + In[i - 1] + In[i - 2]) / 3;
  }
  Serial.println(Out[i]);
  delay(500);
  i++;
}

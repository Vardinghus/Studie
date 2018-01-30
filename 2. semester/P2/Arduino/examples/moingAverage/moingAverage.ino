void setup() {
  Serial.begin(9600);
}

float In[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
float Out[20];
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

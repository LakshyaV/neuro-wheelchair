// Define motor control pins
const int motor1Pin1 = 2; // Pin to IN1 on the L298N
const int motor1Pin2 = 3; // Pin to IN2 on the L298N
const int motor2Pin1 = 4; // Pin to IN3 on the L298N
const int motor2Pin2 = 5; // Pin to IN4 on the L298N

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set all the motor control pins to outputs
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    int prediction = Serial.parseInt(); // Read the incoming integer
    Serial.print("Received prediction: ");
    Serial.println(prediction); // Print the received prediction to the Serial Monitor

    // Perform different actions based on the prediction
    switch (prediction) {
      case 0:
        stopMotors();
        break;
      case 1:
        moveForward();
        break;
      case 2:
        moveBackward();
        break;
      case 3:
        turnLeft();
        break;
      case 4:
        turnRight();
        break;
      default:
        stopMotors();
        break;
    }
  }
}

void moveForward() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

void moveBackward() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

void turnLeft() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

void turnRight() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

void stopMotors() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}

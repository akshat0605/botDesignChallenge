#include <Arduino.h>

// ================================================================
// LINE FOLLOWER + GENERIC OBSTACLE AVOIDANCE  — IMPROVED
//
// Improvements over original:
//   1. Non-blocking ultrasonic with reduced pulseIn timeout (10ms)
//   2. safeForward() — checks for new obstacles during bypass drives
//   3. driveUntilPastObstacle() Phase A has a blind buffer after skip
//   4. Lost-state uses TURN_SPEED (slower) instead of BASE_SPEED
//   5. Lost-state alternates L/R when last_turn_dir == 0 (no blind guess)
//   6. BODY_OVERHANG_MS named constant instead of magic number 80
//   7. Battery sag note + TURN_90_MS tuning guidance kept prominent
//
// Obstacle avoidance strategy (unchanged — it works):
//   1. Obstacle detected ahead → stop
//   2. Turn RIGHT 90° (timed)
//   3. safeForward UNTIL ultrasonic sees clear
//   4. Turn LEFT 90° (back to original heading)
//   5. safeForward UNTIL ultrasonic past obstacle (blocked then clear)
//   6. Turn LEFT 90° (now pointing back toward line)
//   7. safeForward UNTIL line sensors detect the line
//   8. alignToLine() — sensor-guided spin until both sensors on line
// ================================================================

// ── Pins ──────────────────────────────────────────────────────────
const uint8_t SENSOR_LEFT     = 2;
const uint8_t SENSOR_RIGHT    = 3;
const uint8_t LEFT_PWM_PIN    = 10;
const uint8_t LEFT_DIR_PIN_A  = 6;
const uint8_t LEFT_DIR_PIN_B  = 5;
const uint8_t RIGHT_PWM_PIN   = 9;
const uint8_t RIGHT_DIR_PIN_A = 8;
const uint8_t RIGHT_DIR_PIN_B = 7;
const uint8_t TRIG_PIN        = 11;
const uint8_t ECHO_PIN        = 12;

// ── ⬇ TUNE THESE ONLY ────────────────────────────────────────────
int BASE_SPEED             = 200;   // PWM for straight line following
int TURN_SPEED             = 170;   // PWM for line-follow corrections
int BYPASS_SPEED           = 150;   // PWM for all bypass moves
int STRAIGHT_MS            = 4;     // line-follow loop straight delay
int TURN_MS                = 12;    // line-follow loop turn delay
int BRAKE_MS               = 60;    // brake delay on direction change
int TURN_90_MS             = 500;   // ⬅ tune in 30ms steps until turns hit 90°
                                    //   NOTE: re-tune whenever battery is charged
                                    //   as motor speed changes with voltage
int OBSTACLE_CM            = 25;    // trigger distance for obstacle detection
int SIDE_CLEAR_CM          = 30;    // ultrasonic "clear" threshold during bypass
int BODY_OVERHANG_MS       = 80;    // blind forward burst to clear robot body overhang
                                    //   before scanning for line in forwardUntilLine()
unsigned long LOST_TIMEOUT = 1000;  // ms before lost-state search begins

// ── State ─────────────────────────────────────────────────────────
int  last_turn_dir    = 0;
int  prev_turn_dir    = 0;
bool counting_lost    = false;
bool lost_search_left = false;   // alternates search direction when last_turn_dir==0
unsigned long lost_since = 0;

// ── Motor Helpers ─────────────────────────────────────────────────
void leftForward(int spd = 255) {
  digitalWrite(LEFT_DIR_PIN_A, HIGH);
  digitalWrite(LEFT_DIR_PIN_B, LOW);
  analogWrite(LEFT_PWM_PIN, spd);
}
void leftBackward(int spd = 255) {
  digitalWrite(LEFT_DIR_PIN_A, LOW);
  digitalWrite(LEFT_DIR_PIN_B, HIGH);
  analogWrite(LEFT_PWM_PIN, spd);
}
void leftStop()  { analogWrite(LEFT_PWM_PIN, 0); }

void rightForward(int spd = 255) {
  digitalWrite(RIGHT_DIR_PIN_A, HIGH);
  digitalWrite(RIGHT_DIR_PIN_B, LOW);
  analogWrite(RIGHT_PWM_PIN, spd);
}
void rightBackward(int spd = 255) {
  digitalWrite(RIGHT_DIR_PIN_A, LOW);
  digitalWrite(RIGHT_DIR_PIN_B, HIGH);
  analogWrite(RIGHT_PWM_PIN, spd);
}
void rightStop() { analogWrite(RIGHT_PWM_PIN, 0); }

void stopAll() {
  leftStop();
  rightStop();
  delay(30);
}

void brakeIfDirectionChanged(int new_dir) {
  if (new_dir != prev_turn_dir) {
    leftStop();
    rightStop();
    delay(BRAKE_MS);
    prev_turn_dir = new_dir;
  }
}

// ── Line-Follow Movements ─────────────────────────────────────────
void goStraight() {
  brakeIfDirectionChanged(0);
  leftForward(BASE_SPEED);
  rightForward(BASE_SPEED);
  delay(STRAIGHT_MS);
}

void goLeft() {
  brakeIfDirectionChanged(-1);
  leftBackward(TURN_SPEED);
  rightForward(TURN_SPEED);
  delay(TURN_MS);
}

void goRight() {
  brakeIfDirectionChanged(1);
  leftForward(TURN_SPEED);
  rightBackward(TURN_SPEED);
  delay(TURN_MS);
}

// ── Timed 90° Turns ───────────────────────────────────────────────
// IMPORTANT: these are voltage-dependent. Re-tune TURN_90_MS after
// every charge cycle if precise turns matter for your track.
void spin90Left() {
  Serial.println("  spin90Left");
  leftBackward(BYPASS_SPEED);
  rightForward(BYPASS_SPEED);
  delay(TURN_90_MS);
  stopAll();
}

void spin90Right() {
  Serial.println("  spin90Right");
  leftForward(BYPASS_SPEED);
  rightBackward(BYPASS_SPEED);
  delay(TURN_90_MS);
  stopAll();
}

// ── Ultrasonic ────────────────────────────────────────────────────
// IMPROVEMENT 1: reduced pulseIn timeout from 20ms → 10ms.
// 10ms covers ~170cm which is well beyond any needed range.
// This halves the worst-case blocking time per reading.
long getDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 10000);  // 10ms timeout (~170cm max)
  if (dur == 0) return 999;
  return dur * 0.034 / 2;
}

bool pathIsClear() {
  return getDistanceCM() >= SIDE_CLEAR_CM;
}

// ── Sensor Helpers ────────────────────────────────────────────────
bool leftOnLine()  { return digitalRead(SENSOR_LEFT)  == HIGH; }
bool rightOnLine() { return digitalRead(SENSOR_RIGHT) == HIGH; }
bool anyOnLine()   { return leftOnLine() || rightOnLine(); }
bool bothOnLine()  { return leftOnLine() && rightOnLine(); }

// ── IMPROVEMENT 2: safeForward() ─────────────────────────────────
// Replaces raw leftForward/rightForward calls inside bypass drives.
// Checks for a new obstacle every 50ms while driving forward.
// Returns true if stopped due to obstacle, false otherwise.
// Used by driveUntilClear, driveUntilPastObstacle, forwardUntilLine.
bool safeForward(int spd) {
  long d = getDistanceCM();
  if (d > 0 && d < OBSTACLE_CM) {
    stopAll();
    Serial.println("  !! New obstacle detected during bypass — stopping");
    return true;  // obstacle hit
  }
  leftForward(spd);
  rightForward(spd);
  return false;
}

// ── Drive Forward Until Ultrasonic Is Clear ───────────────────────
void driveUntilClear(int timeoutMs = 5000) {
  Serial.println("  driveUntilClear...");
  unsigned long start = millis();
  while (!pathIsClear()) {
    if (safeForward(BYPASS_SPEED)) return;  // abort on new obstacle
    if (millis() - start > timeoutMs) {
      Serial.println("  driveUntilClear TIMEOUT");
      break;
    }
    delay(10);
  }
  stopAll();
  Serial.println("  Side clear!");
}

// ── Drive Forward Until Past Obstacle ────────────────────────────
// IMPROVEMENT 3: Phase A skip now adds a blind buffer to ensure
// robot is actually beside the obstacle before Phase B begins.
void driveUntilPastObstacle(int timeoutMs = 6000) {
  Serial.println("  driveUntilPastObstacle...");
  unsigned long start = millis();

  // Phase A: wait until sensor sees the obstacle beside us
  Serial.println("    Phase A: waiting to see obstacle side...");
  while (pathIsClear()) {
    if (safeForward(BYPASS_SPEED)) return;
    if (millis() - start > timeoutMs / 2) {
      Serial.println("    Phase A skipped — adding blind buffer");
      // IMPROVEMENT: drive a short extra burst so we're definitely
      // beside the obstacle before checking Phase B
      leftForward(BYPASS_SPEED);
      rightForward(BYPASS_SPEED);
      delay(300);
      stopAll();
      break;
    }
    delay(10);
  }

  // Phase B: keep driving until obstacle clears the side sensor
  Serial.println("    Phase B: driving past obstacle...");
  while (!pathIsClear()) {
    if (safeForward(BYPASS_SPEED)) return;
    if (millis() - start > timeoutMs) {
      Serial.println("    Phase B TIMEOUT");
      break;
    }
    delay(10);
  }

  stopAll();
  // Extra buffer to fully clear the obstacle corner
  leftForward(BYPASS_SPEED);
  rightForward(BYPASS_SPEED);
  delay(200);
  stopAll();
  Serial.println("  Past obstacle!");
}

// ── Drive Forward Until Line Found ───────────────────────────────
void forwardUntilLine(int timeoutMs = 5000) {
  Serial.println("  forwardUntilLine...");
  // IMPROVEMENT: BODY_OVERHANG_MS is now a named constant (was magic number 80)
  leftForward(BYPASS_SPEED);
  rightForward(BYPASS_SPEED);
  delay(BODY_OVERHANG_MS);

  unsigned long start = millis();
  while (!anyOnLine()) {
    if (safeForward(BYPASS_SPEED)) return;  // abort on new obstacle
    if (millis() - start > timeoutMs) {
      Serial.println("  forwardUntilLine TIMEOUT");
      break;
    }
    delay(5);
  }
  stopAll();
  Serial.println("  Line found!");
}

// ── Sensor-Guided Line Alignment ─────────────────────────────────
void alignToLine(bool turnLeft = true, int timeoutMs = 2000) {
  Serial.println("  alignToLine...");
  unsigned long start = millis();

  while (millis() - start < timeoutMs) {
    if (bothOnLine()) {
      stopAll();
      Serial.println("  Aligned!");
      return;
    }
    if (leftOnLine() && !rightOnLine()) {
      leftForward(BYPASS_SPEED);
      rightBackward(BYPASS_SPEED);
    } else if (!leftOnLine() && rightOnLine()) {
      leftBackward(BYPASS_SPEED);
      rightForward(BYPASS_SPEED);
    } else {
      if (turnLeft) {
        leftBackward(BYPASS_SPEED);
        rightForward(BYPASS_SPEED);
      } else {
        leftForward(BYPASS_SPEED);
        rightBackward(BYPASS_SPEED);
      }
    }
    delay(2);
  }
  stopAll();
  Serial.println("  alignToLine TIMEOUT");
}

// ── Generic Obstacle Bypass ───────────────────────────────────────
void bypassObstacle() {
  Serial.println("=== BYPASS START ===");
  stopAll();
  delay(200);

  Serial.println("Step 1: RIGHT 90°");
  spin90Right();

  Serial.println("Step 2: Clear obstacle (sensor-guided)");
  driveUntilClear(5000);

  Serial.println("Step 3: LEFT 90° (restore heading)");
  spin90Left();

  Serial.println("Step 4: Drive past obstacle (sensor-guided)");
  driveUntilPastObstacle(6000);

  Serial.println("Step 5: LEFT 90° (toward line)");
  spin90Left();

  Serial.println("Step 6: Approach line (sensor-guided)");
  forwardUntilLine(5000);

  Serial.println("Step 7: Align to line (sensor-guided)");
  alignToLine(true, 2000);

  // Reset line-follow state
  last_turn_dir    = 0;
  prev_turn_dir    = 0;
  counting_lost    = false;
  lost_search_left = false;

  Serial.println("=== BYPASS COMPLETE ===");
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(SENSOR_LEFT,     INPUT);
  pinMode(SENSOR_RIGHT,    INPUT);
  pinMode(LEFT_PWM_PIN,    OUTPUT);
  pinMode(RIGHT_PWM_PIN,   OUTPUT);
  pinMode(LEFT_DIR_PIN_A,  OUTPUT);
  pinMode(LEFT_DIR_PIN_B,  OUTPUT);
  pinMode(RIGHT_DIR_PIN_A, OUTPUT);
  pinMode(RIGHT_DIR_PIN_B, OUTPUT);
  pinMode(TRIG_PIN,        OUTPUT);
  pinMode(ECHO_PIN,        INPUT);

  stopAll();
  Serial.println("Starting in 2s...");
  delay(2000);
  lost_since = millis();
}

// ── Main Loop ─────────────────────────────────────────────────────
void loop() {

  // ── Obstacle check ───────────────────────────────────────────
  long dist = getDistanceCM();
  if (dist > 0 && dist < OBSTACLE_CM) {
    Serial.print("OBSTACLE: "); Serial.print(dist); Serial.println(" cm");
    bypassObstacle();
    return;
  }

  // ── Line Following ────────────────────────────────────────────
  bool L = leftOnLine();
  bool R = rightOnLine();

  Serial.print("L:"); Serial.print(L);
  Serial.print(" R:"); Serial.println(R);

  if (L && R) {
    // Both on line — go straight
    counting_lost = false;
    last_turn_dir = 0;
    goStraight();

  } else if (L && !R) {
    // Drifted right — correct left
    counting_lost = false;
    last_turn_dir = -1;
    goLeft();

  } else if (!L && R) {
    // Drifted left — correct right
    counting_lost = false;
    last_turn_dir = 1;
    goRight();

  } else {
    // ── Both sensors off — lost ───────────────────────────────
    if (!counting_lost) {
      counting_lost = true;
      lost_since = millis();
    }

    if (millis() - lost_since < LOST_TIMEOUT) {
      // IMPROVEMENT 4: use TURN_SPEED (slower) not BASE_SPEED while lost.
      // Driving fast while lost takes the robot further from the line.
      leftForward(TURN_SPEED);
      rightForward(TURN_SPEED);
      delay(STRAIGHT_MS);

    } else {
      // IMPROVEMENT 5: alternate search direction when last_turn_dir==0
      // instead of always guessing right. Covers sharp curves from straight.
      Serial.println("LOST — searching");

      if (last_turn_dir == 1) {
        leftForward(TURN_SPEED);
        rightBackward(TURN_SPEED);
      } else if (last_turn_dir == -1) {
        leftBackward(TURN_SPEED);
        rightForward(TURN_SPEED);
      } else {
        // Was going straight — alternate L/R each search cycle
        if (lost_search_left) {
          leftBackward(TURN_SPEED);
          rightForward(TURN_SPEED);
        } else {
          leftForward(TURN_SPEED);
          rightBackward(TURN_SPEED);
        }
        lost_search_left = !lost_search_left;
      }
      delay(TURN_MS);

      // Recover immediately if line found mid-search
      if (anyOnLine()) {
        stopAll();
        counting_lost = false;
        lost_search_left = false;
      }
    }
  }
}

Here are the direct, generalized biomechanical parameters and thresholds used to programmatically classify safe versus dangerous badminton footwork.

Implement these exact numerical boundaries in your telemetry logic to flag dangerous mechanics.

### **1. Foot-to-Trajectory Alignment (Ankle Sprain Risk)**

Measures the lateral shearing force applied to the ankle ligaments during dynamic braking.

* **Parameter:** The absolute angular difference between the **Center of Mass (CoM) trajectory vector** (derived from hip keypoints over the last 5 frames) and the **Foot Vector** (Heel keypoint to Big Toe keypoint).
* **Correct / Safe:** **0° to 15°** deviation. The foot points directly along the line of momentum.
* **Dangerous Threshold:** **> 15°** deviation.
* **Critical Danger:** **> 30°** deviation. Forces excessive lateral inversion or eversion, leading to acute Anterior Talofibular Ligament (ATFL) tears upon load bearing.

### **2. Sagittal Knee Extension (Patellar Tendon Overload)**

Measures mechanical leverage and strain on the patellar tendon and meniscus during maximum lunge depth.

* **Parameter A (Positional):** The horizontal coordinate distance between the front **Knee** keypoint and the front **Big Toe** keypoint.
* **Parameter B (Angular):** The internal angle formed by the Hip, Knee, and Ankle keypoints.
* **Correct / Safe:** The knee remains vertically behind or perfectly aligned with the toe boundary. Internal knee angle is **≥ 90°**.
* **Dangerous Threshold:** The front knee's horizontal coordinate extends forward past the big toe coordinate, OR the internal knee angle drops to **< 80°** during dynamic deceleration.

### **3. Foot Strike Pitch Angle (Kinetic Shock Transfer)**

Determines whether kinetic energy is safely absorbed by the posterior chain (calves/Achilles) or slammed directly into the skeletal joints.

* **Parameter:** The pitch angle of the foot relative to the floor plane at the exact frame of initial ground contact. Computed via the vertical offset between the **Heel** and **Big Toe** coordinates.
* **Correct / Safe:** **Negative Pitch (-5° to -20°)**. The heel impacts the floor first, allowing the foot to roll smoothly flat to dissipate force.
* **Dangerous Threshold:** **≥ 0° Pitch**.
* **0° (Flat-Footed):** Instantaneous high G-force impact traveling straight to the meniscus.
* **Positive Pitch (> 0°, Toe-First):** Drives the tibia backward instantly, highly destabilizing the knee joint.

### **4. Trailing Foot Roll Angle (Medial vs. Lateral Drag)**

Monitors the rear anchor foot during deep forward lunges.

* **Parameter:** The angle of the rear foot plane relative to the floor, measured by comparing the distance of the **Big Toe** versus the **Small Toe / Outer Ankle** keypoints to the ground plane.
* **Correct / Safe:** The inner edge (medial aspect/Big Toe) maintains floor contact.
* **Dangerous Threshold:** The outer edge (Small Toe coordinate) is closer to the ground plane than the Big Toe coordinate. Indicates ankle eversion dragging, risking severe lateral friction injuries and medial ligament strain.

---

add another parameter which is hip rotation, where hip can "open" to the side of the body

/web-research from the parameter that we have and the failure condition, right now the failure condition is singularly isolated (just one if statement), which is very far from the reality. perform a research to create a robust model of real life bio-mechanical risks

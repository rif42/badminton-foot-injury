# Biomechanical Parameter Constraints

## Injury Risk Model (from `injury.md`)

The project classifies badminton footwork as **safe** or **dangerous** based on five biomechanical parameters, each with a "correct" range and a dangerous threshold. All thresholds are derived from published research.

### 1. Foot-to-Trajectory Alignment (Ankle Sprain Risk)
- **Parameter**: Absolute angular difference between CoM trajectory vector and foot vector
- **Safe**: ≤ 15° deviation
- **Critical danger**: > 30° deviation → ATFL tear risk

### 2. Sagittal Knee Extension (Patellar Tendon Overload)
- **Parameter A**: Horizontal distance front knee to big toe
- **Parameter B**: Internal angle Hip-Knee-Ankle
- **Safe**: Front knee ≤ big toe horizontally AND internal knee angle ≥ 90°
- **Danger threshold**: Front knee extends past big toe OR internal angle < 80°

### 3. Foot Strike Pitch Angle (Kinetic Shock Transfer)
- **Parameter**: Vertical offset between Heel and Big Toe at initial contact
- **Safe**: −5° to −20° pitch (heel-first strike, posterior chain absorbs force)
- **Danger threshold**: ≥ 0° pitch (flat-footed or toe-first → direct skeletal impact)

### 4. Trailing Foot Roll Angle (Medial vs. Lateral Drag)
- **Parameter**: Relative distance of Big Toe to Small Toe/Outer Ankle keypoints from ground plane
- **Safe**: Medial aspect maintains floor contact
- **Danger threshold**: Outer edge closer to ground than big toe → ankle eversion dragging

### 5. Hip Rotation (Hip Opening Control) — Proposed Addition
- Monitors whether the hip "opens" laterally during movement
- Intended as a proximal-to-distal risk factor coupling with knee valgus mechanics

## Threshold Implementation Rules
- Every biomechanical parameter must be computed from MediaPipe landmarks, never hardcoded.
- Parameters are independent measurements — each can flag dangerous mechanics on its own, but the **combined risk model** (future work) will weigh them together.
- All threshold values in `injury.md` are treated as authoritative references; any change requires citation of updated research evidence.

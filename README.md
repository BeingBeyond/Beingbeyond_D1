<p align="center">
  <img src="bb_d1.png" width="400" alt="BeingBeyond D1">
</p>

# BeingBeyond D1 SDK Examples

This repository provides the BeingBeyond D1 SDK, example Python scripts, and basic guidance for environment setup and common first-time troubleshooting.

> **WARNING**  
> Keep the emergency stop button within reach at all times.

---

## 1. Requirements

### 1.1 Hardware
- BeingBeyond D1 robot  
  - Head + arm  
  - DexHand  
  - Intel RealSense RGB-D camera  
- Linux PC (Ubuntu 20.04 / 22.04) or Windows (coming soon)  
- USB 3.0 port  

### 1.2 Software
- Python 3.8 or 3.10  
- pyrealsense2
  - `pip install pyrealsense2`
- Pre-built SDK wheels in `/lib` (choose according to your Python version):  
  - `beingbeyond_d1_sdk-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`  
  - `beingbeyond_d1_sdk-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`  
- Windows wheels will be added in future releases.

---

## 2. Installation

### 2.1 Create Conda environment

```bash
conda create -n bb_d1 python=3.8 -y
conda activate bb_d1
```

### 2.2 Install the SDK wheel

```bash
pip install -U pip
pip install lib/beingbeyond_d1_sdk-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 2.3 Install additional dependencies

```bash
pip install numpy pyrealsense2 opencv-python
```

---

## 3. USB & Permissions (Important)

### 3.1 Check USB serial device

```bash
sudo apt remove brltty
```

Reinsert the USB cable.

```bash
ls /dev/ttyUSB*
```
Expected: `/dev/ttyUSB0` or `/dev/ttyUSB1`.


### 3.2 If you get “permission denied”

```bash
groups
sudo usermod -a -G dialout $USER
```

Log out and re-login.

---

## 4. Hardware Setup

1. Power on the robot; wait for DexHand auto-calibration.  
2. Keep E-STOP reachable.  
3. If abnormal motion occurs:

   press E-STOP → stop scripts → restore pose → release E-STOP → retry.  
4. If calibration fails: 

    press E-STOP → release E-STOP → retry

---

## 5. Running the Examples

```bash
cd examples
```

### 5.1 DexHand  
```bash
python 1_control_hand.py
```

### 5.2 Head + Arm  
```bash
python 2_control_head_arm.py
```

### 5.3 RealSense viewer  
```bash
pip install pyrealsense2
python 3_show_vision.py
```

### 5.4 Full D1 Demo  
```bash
python 4_control_d1.py
```

### 5.5 IK to Target EE Pose

```bash
python 5_ik_control.py
```

This script demonstrates:

- Querying the current end-effector pose using **D1Kinematics**
- Building a target pose in the base frame  
- Running iterative **arm-only IK**
- Inspecting how small Cartesian offsets map to joint motions


---

### 5.6 Keyboard Teleoperation

```bash
python 6_keyboard_teleop.py
```

> **IMPORTANT**  
> This example **must be run inside a real terminal** (Ubuntu Terminal / macOS Terminal).  
> Better DO NOT run it in VSCode, PyCharm, Jupyter, or other IDEs — raw-keyboard mode will NOT work.

This script provides Real-time Cartesian teleoperation of the D1 arm using the keyboard

##### Keyboard Commands

```
Translation:
  w / s : X+ / X-
  a / d : Y+ / Y-
  z / x : Z+ / Z-

Orientation:
  u / o : roll  + / -
  i / k : pitch + / -
  j / l : yaw   + / -

Hand Control:
  Space : toggle hand_pos between 0 and 1

Other:
  r     : reset EE target and joint state
  h     : print help
  q     : quit
```

This example is the recommended entry point for **interactive D1 teleoperation**.

---

## 6. Troubleshooting (Quick)

- **No /dev/ttyUSB***  
  - Reinsert cable  
  - Remove brltty  
  - Check `lsusb`

- **Permission denied**  
  - Add user to `dialout`

- **CAN not up**  
  ```bash
  sudo ip link set can0 up type can bitrate 1000000
  ```

- **Hand misaligned**  
  - Power cycle and wait for calibration

---

## 7. License
MIT / Apache-2.0

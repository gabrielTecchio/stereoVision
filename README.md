# ! UNDER CONSTRUCTION !
# Stereo Vision for a Robotic Manipulator
## Overview
This code was developed to compare two approaches for the parallax phenomenon: a traditional calculation and a neural network model. The academic article is available at the following URL: [LUME | UFRGS (pt-BR)](https://lume.ufrgs.br/handle/10183/280408)[^1].

The Python script running in my computer communicates with an EPSON SCARA robot (4 DoF) via TCP/IP protocol to manipulate a rectangular object.

![Elements configuration](/assets/images/elementsConfiguration.JPG)

## Features
* Traditional Stereo Vision
* Neural Network Model for Stereo Vision
* Parallax Phenomenon Analysis
* Real-time Robot Communication via TCP/IP Protocol
* Robotic Manipulation Capabilities
* 4 Degrees of Freedom (DoF) Pose Estimation
* Integration with EPSON SPEL RC+ 5.0

## Prerequisites:
1. Python Installation: Ensure you have Python installed on your machine.
2. EPSON Robots and RC +5.0: This project is designed to work with EPSON robots via the RC +5.0 software. Make sure to check and configure TCP/IP settings to establish communication between your computer and the robot controller.
3. Dependencies: To install the necessary dependencies, run:
```python
pip install -r requirements.txt
```
4. Setup and Execution: Set up the cameras and run the main function with:
```python
python main.py
```

## Collaboration & Inquiries
Feel free to reach out if you have any questions about the project or if you're interested in collaborating. I'm always open to feedback and eager to connect with fellow enthusiasts!

[^1]: References of this study could be found in the academic article as well as the program graphcet.
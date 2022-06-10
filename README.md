Welcome to PhoenixBot!
======================

PhoenixBot is the computer vision software for autonomous EV charging.

Prerequisites:
=================

Requirements:
-----------------
 - Python 3.8
 - CUDA 11.3

Equipment:
-----------------
 - Two (or at least one) UR robots. Preferably UR10 and UR3.
 - Any intel realsense camera.
 - Charging plug and socket of Type 2 (EU).
 - 3D printed rim, imitating a part of a car (can be found in _print_)

To run experiment:
=================

1. Fine-tune the model. To set the TCP with respect to model systematic error.

2. Set camera calibration parameters. Use any external tool for that.

3. Set views for rough pose estimation.

Model fine-tuning:
-----------------
For model fine-tuning:

<pre>
python finetune.py
</pre>

Follow the instruction on the screen.\
Parameters to be sent to finetune.py: 
> -e:  (int) to fix camera exposure,\
> -rndori: (bool) to randomize camera orientations during fine-tuning\
> -camopt: (bool) to optimize camera calibration parameters\
> -d: (bool) debug mode


Running the experiment:
----------------
To run the experiment: 
<pre>
python insertion_test.py
</pre>

The team:
=================
Viacheslav Martynov\
Viktor Rakhmatullin



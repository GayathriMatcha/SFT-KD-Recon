# SFTN-KD-Recon
## Learning a Student-friendly Teacher for Knowledge Distillation in MRI Reconstruction
The base code is taken from https://github.com/Bala93/KD-MRI

### DATASET:
[Automated Cardiac Diagnosis Challenge (ACDC)](https://ieeexplore.ieee.org/document/8360453)

![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/sft_overview.png?raw=true)
Student-Friendly training of the teacher. The teacher DC-CNN has five blocks, each having CNN with five convolution layers and DF layer, and the student DC-CNN has five blocks, each having three convolution layers and a DF layer. The teacher is trained with three loss terms. Note that all the blocks of the student learn initial weights except the first block during SFT training.
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/SFT_teacher.png?raw=true)
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/algorithm.png?raw=true)
## Results for MRI Reconstruction: 
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/results.png?raw=true)
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/Quant_result.png?raw=true)



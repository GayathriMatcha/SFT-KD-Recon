# SFTN-KD-Recon
## Learning a Student-friendly Teacher for Knowledge Distillation in MRI Reconstruction
The base code is taken from https://github.com/Bala93/KD-MRI


### KNOWLEDGE DISTILLATION:
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/sft_overview.png?raw=true)

### Student-Friendly training of the teacher :

The teacher DC-CNN has five blocks, each having CNN with five convolution layers and DF layer, and the student DC-CNN has five blocks, each having three convolution layers and a DF layer. The teacher is trained with three loss terms. Note that all the blocks of the student learn initial weights except the first block during SFT training.

![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/SFT_teacher.png?raw=true)
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/algorithm.png?raw=true)
### DATASET:
1. [Automated Cardiac Diagnosis Challenge (ACDC)](https://ieeexplore.ieee.org/document/8360453)
2. [MRBrainS dataset](https://www.hindawi.com/journals/cin/2015/813696/)
### Results for MRI Reconstruction: 
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/results.png?raw=true)
### Qualitative Results for MRI Reconstruction:
Visual results (from left to right): target, target inset, ZF, teacher, student, Std-KD, SFT-KD-Recon, student residue, Std-KD residue, SFT-KD-Recon residue with respect to the target, for the brain (top) and cardiac (bottom) with 4x acceleration. We note that in addition to lower reconstruction errors, the SFT-KD distilled student is able to retain finer structures better when compared to the student and Std-KD output.
![alt text](https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/Quant_result.png?raw=true)



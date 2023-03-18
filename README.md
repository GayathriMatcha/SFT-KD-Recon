# SFTN-KD-Recon
## Learning a Student-friendly Teacher for Knowledge Distillation in MRI Reconstruction
The base code is taken from https://github.com/Bala93/KD-MRI


## MRI Reconstruction:
## Knowledge Distillation:
Student-Friendly training of the teacher. The teacher DC-CNN has five blocks,
each having CNN with five convolution layers and DF layer, and the student DC-CNN has
five blocks, each having three convolution layers and a DF layer. The teacher is trained
with three loss terms - $$L^T_{rec}, LS
rec (blue arrows), and Limit (violet arrows). Note that all the
blocks of the student learn initial weights except the first block during SFT training.

https://github.com/GayathriMatcha/SFTN-KD-Recon/blob/main/Images/Quant_result.png




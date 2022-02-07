# A Fusion architecture for Human Activity Recognition

## About:-

This work has been accepted at IEEE [INDICON](https://www.ewh.ieee.org/r10/calcutta/indicon2021/index.html) 2021! It was carried out under the supervision of Professor [Amlan Chakrabarti](https://sites.google.com/caluniv.ac.in/amlanc-org/), Department of AKCSIT, University of Calcutta.

The paper is available [here](https://doi.org/10.1109/INDICON52576.2021.9691648) and the slides can be viewed [here](https://sarosijbose.github.io/files/INDICON_presentation_2021.pdf).

## Setup:-  

A. Clone the repository first.  
```bash
git clone https://github.com/sarosijbose/A-Fusion-architecture-for-Human-Activity-Recognition.git
```

B. It is then recommended to create a fresh virtual environment.
```bash
python -m venv env
source activate env/bin/activate
```
Then install the required dependencies.
```bash
pip install -r requirements.txt
```
C. Directory structure overview  
The codebase is divided into 5 folders. 

1. 3D CNN  
This folder contains all the code necessary for running the Spatial 3D CNN Stream.
First convert the sample UCF-101 videos given in the ```sample videos``` folder into their required pre-processed format,
```bash
python pre_processing.py
```
This will convert the videos into the required .npy format.
Next, feed them *one-by-one* into the evaluation code for the results
```bash
python eval3dcnn.py
```
Make sure that the ```i3d.py``` file is present in the same directory and change the checkpoint path accordingly.
The Kinetics-RGB-600 checkpoint is avaliable [here](https://drive.google.com/drive/folders/1bLwYRzp7Aei1qtNhOcq5C4cnjD27A845?usp=sharing).

The top 5-predictions will be printed in the order of their prediction accuracy.
Here is a sample output for the video ```v_BrushingTeeth_g17_c02.npy```  

![Sample Output](https://github.com/sarosijbose/An-ensemble-architecture-for-Human-Activity-Recognition/blob/main/3D%20CNN/sample_output.jpg)

2. 2D CNN  
This folder contains the code for running the Spatial 2D CNN Stream.

Next, directly feed the frames by running this,
```bash
python eval2dcnn.py
```
Here is a sample output:-
![Sample Output](https://github.com/sarosijbose/An-ensemble-architecture-for-Human-Activity-Recognition/blob/main/2D%20CNN/Sample_output_2dcnn.jpg)
Download the entire pre-processed RGB data made avaliable by Feichtenhofer [here](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/) if you want to fine-tune on UCF-101.

3. Average the softmax scores of each stream.
```bash
python average_fusion.py
```
4. Data   
This folder contains all the required utilities and samples required for evaluation.

## Citation:-

Please consider citing this work if you found it useful:-

```latex
@INPROCEEDINGS{9691648,
  author={Bose, Sarosij and Chakrabarti, Amlan},
  booktitle={2021 IEEE 18th India Council International Conference (INDICON)}, 
  title={A Fusion Architecture Model for Human Activity Recognition}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/INDICON52576.2021.9691648}}
```

## Acknowledgements:-

Parts of the codebase and the RGB-600 checkpoint have been adapted from the [Kinetics](https://github.com/deepmind/kinetics-i3d) repository. 
We are grateful to the authors for making their work available.

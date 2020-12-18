# Dahu CNN Inside-Outside-Guidance (IOG)
This project hosts the code for the Dahu CNN IOG algorithms for interactive segmentation.


The code will be released soon. Please stay tuned.



<!-- ### Demo

<table>
    <tr>
        <td width="30%">
	<img src="https://github.com/shiyinzhang/Inside-Outside-Guidance/blob/master/ims/IOG.gif"/>
        </td>   
        <td width="30%">
	<img src="https://github.com/shiyinzhang/Inside-Outside-Guidance/blob/master/ims/refinement.gif"/>
        </td>   
        <td width="30%">
	<img src="https://github.com/shiyinzhang/Inside-Outside-Guidance/blob/master/ims/cross_domain.gif"/>
        </td> 
    </tr>
    <tr>
        <td width="30%" align="center">
	IOG(3 points)
        </td>   
        <td width="30%" align="center">
	IOG(Refinement)
        </td>   
        <td width="30%" align="center">
	IOG(Cross domain)
        </td> 
    </tr>
</table> -->


### Installation
1. Install requirement  
  - PyTorch = 0.4
  - python >= 3.5
  - torchvision = 0.2
  - pycocotools
2. Usage
You can start training with the following commands:
```
# training step
python train.py

# testing step
python test.py

# train step
python eval.py
```
We set the paths of PASCAL/SBD dataset and pretrained model in mypath.py.

### Pretrained models
| Dataset | Backbone |      Download Link        |
|---------|-------------|:-------------------------:|
|PASCAL + SBD  |  ResNet-101 |  [IOG_PASCAL_SBD.pth](https://drive.google.com/file/d/1Lm1hhMhhjjnNwO4Pf7SC6tXLayH2iH0l/view?usp=sharing)     |
|PASCAL |  ResNet-101   |  [IOG_PASCAL.pth](https://drive.google.com/file/d/1GLZIQlQ-3KUWaGTQ1g_InVcqesGfGcpW/view?usp=sharing)   |






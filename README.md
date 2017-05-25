# cntk-cyclegan
CNTK based implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[[paper]](https://arxiv.org/pdf/1703.10593.pdf)

# How-to run the code
1) Yosemity dataset [Yosemity dataset](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip) is expected to be unzipped into  ./data folder
2) dataUtils.py generates map files for input
3) If you ran on GPU and see out-of-memory exception => lower batch size

# Results
I have ran trainCycleGan.py on [Yosemity dataset](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip) and batch size 4. This dataset is not super clean, the set of summer imagages has several winter images and vice versa. I did quick clean up of those before training.
Also I noticed that in current implementation I have G(X) that transfers summer Yosemity to winter works better than F(X) (winter to summer). Also Generator tends to change daytime to evening\night time.

<img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/45800_realX_bgr.png" width="128px"/> <img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/45800_bgr.png" width="128px"/>

<img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/30000_realX_bgr.png" width="128px"/> <img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/30000_bgr.png" width="128px"/>

<img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/30200_realy_bgr.png" width="128px"/> <img src="https://raw.githubusercontent.com/olgaliak/cntk-cyclegan/master/imgs/30200_F_bgr.png" width="128px"/>



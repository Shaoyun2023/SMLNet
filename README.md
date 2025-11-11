🧩🚀SMLNet: A SPD Manifold Learning Network for Infrared and Visible Image Fusion
====
Accetped by 
[IJCV 2025]
🔗"(https://doi.org/10.1007/s11263-025-02578-1)"

🔍Highlights
---
● **First SPD Manifold Learning for Fusion**  
Our work is the first to introduce Riemannian manifold networks (SPD manifolds) into image fusion tasks, enabling geometrically consistent modeling of cross-modal correlations.

● **Manifold-Aware Attention**  
We propose a novel SPD Attention Module (SPDAM) that dynamically weights cross-modal features on the manifold space, enhancing complementary information fusion while suppressing redundancies. 

● **Superior Performance & Efficiency**  
Extensive experiments show SMLNet outperforms state-of-the-art methods in fusion quality (e.g., EN, VIF) and computational efficiency, with proven gains in downstream tasks like object detection.

🖥️Environment
----
python==3.12.7

pytorch==2.5.1

pytorch-cuda==12.4

scipy==1.13.1

numpy==1.26.4

pillow==10.4.0

tqdm==4.66.5

⚙️Training
----
```bash
python train_autoencoder.py
```

✔️Testing
----
```bash
python test.py
```

📖Citation
----
If you are interested in our work, please cite it in the following format:
```bash
@article{kang2025smlnet,
  title={SMLNet: A SPD Manifold Learning Network for Infrared and Visible Image Fusion},
  author={Kang, Huan and Li, Hui and Xu, Tianyang and Wu, Xiao-Jun and Wang, Rui and Cheng, Chunyang and Kittler, Josef},
  journal={International Journal of Computer Vision},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```

⬇️Model Download
----
The vgg16 model can be found in https://pan.baidu.com/s/14YYYrDZ1RM3yqFbYNnbQbw, and the password is: usd6

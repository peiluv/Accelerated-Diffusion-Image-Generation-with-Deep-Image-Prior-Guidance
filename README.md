# 《FUNDAMENTALS AND APPLICATIONS OF GENERATIVE AI》
- Generative Models for Visual Signals– Assignment
- 2024/6/17

## **Overview**

　　Diffusion models have emerged as strong contenders to GANs in image generation tasks
due to their superior sample quality and diversity. One of the most prominent models in this category is the
Denoising Diffusion Probabilistic Model (DDPM), which reconstructs target images by progressively
removing Gaussian noise. While DDPMs exhibit stable and diverse generation behavior, a notable drawback
remains: they rely on pure Gaussian noise as the initial input, which is semantically unaligned with the target
image. This often results in prolonged inference time and high computational costs, limiting scalability and
practical deployment.

　　To address these limitations, this project introduces a hybrid generation strategy that integrates Deep
Image Prior (DIP) as a semantically informed initial condition for DDPMs. DIP is an unsupervised image
generation method that leverages the structure of convolutional neural networks to learn high-level features
(e.g., contours, textures, and edges) from a target image using only random noise as input. This makes DIP an
ideal candidate for generating initial images that are closer in distribution to the desired output.

---

## **Method**

![image](https://github.com/user-attachments/assets/2bce19a5-e667-4652-80e5-9d682eec426a)

---

## **File structure**

```
  project/
  ├── main.py
  ├── config.py
  ├── models/
  │   ├── ema.py
  │   ├── ddpm.py
  │   ├── dip.py
  │   └── integration.py
  ├── utils/
  │   ├── dataset.py
  │   ├── metrics.py
  │   └── visualization.py
  └── experiments/
     ├── train.py
     ├── pretrain_dip.py
     ├── pretrain_baseline_ddpm.py
     ├── pretrain_integrated_ddpm05.py
     └── pretrain_integrated_ddpm10.py
```

---

## **result**

|Method|	Number of FID Samples	|Reverse Steps|	FID	|LPIPS	|PSNR	|SSIM|	Inference Time|
|-|-|-|-|-|-|-|-|
|DDPM	|1024	|1000|	58.2	|0.47|	9.6|	0.17|	42.4 s|
|DIP-Guided DDPM	|1024	|**200**|	53.07	|0.49	|9.15	|0.20	|**0.0083 s**|

![dip_gif](https://github.com/user-attachments/assets/36027e70-e1c4-41a9-b588-01a2acd5c273)

![dip_gif](https://github.com/user-attachments/assets/ec7ad4a8-e44c-470c-956b-fd8ac1e401ed)

![dip_gif](https://github.com/user-attachments/assets/8eed1948-cef7-47ef-b257-0cb02b675ed9)

---

## **References**

1. [Deep Diffusion Image Prior for Efficient OOD Adaptation in 3D Inverse Problems (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09637.pdf)
2. [MetaDIP: Accelerating Deep Image Prior with Meta Learning (arXiv)](https://openaccess.thecvf.com/content/CVPR2023/papers/Fei_Generative_Diffusion_Prior_for_Unified_Image_Restoration_and_Enhancement_CVPR_2023_paper.pdf)
3. [Generative Diffusion Prior for Unified Image Restoration and Enhancement (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Fei_Generative_Diffusion_Prior_for_Unified_Image_Restoration_and_Enhancement_CVPR_2023_paper.pdf)



<div align="center">

# FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion

[Pengwei Liang](https://scholar.google.com/citations?user=54Ci0_0AAAAJ&hl=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ), [Qing Ma](https://scholar.google.com/citations?user=x6QQGQkAAAAJ&hl=en), [Chenyang Wang](https://scholar.google.com/citations?user=yMW-xMgAAAAJ), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ)

Harbin Institute of Technology, Harbin 150001, China. Electronic Information School, Wuhan University, Wuhan 430072, China.

</div>

## [Paper (IEEE TIP 2025)]([https://ieeexplore.ieee.org/document/11114795/](https://ieeexplore.ieee.org/document/11114795/)) 

> Infrared images exhibit a significantly different appearance compared to visible counterparts. Existing infrared and visible image fusion (IVF) methods fuse features from both infrared and visible images, producing a new “image” appearance not inherently captured by any existing device. From an appearance perspective, infrared, visible, and fused images belong to different data domains. This difference makes it challenging to apply fused images because their domain-specific appearance may be difficult for downstream systems, e.g., pretrained segmentation models. Therefore, accurately assessing the quality of the fused image is challenging. To address those problem, we propose a novel IVF method, FusionINV, which produces fused images with an appearance similar to visible images. FusionINV employs the **pre-trained Stable Diffusion (SD)** model to invert infrared images into the noise feature space. To inject visible-style appearance information into the infrared features, we leverage the inverted features from visible images to guide this inversion process. In this way, we can embed all the information of infrared and visible images in the noise feature space, and then use the prior of the pre-trained SD model to generate visually friendly images that align more closely with the RGB distribution. Specially, to generate the fused image, we design a tailored fusion rule within the denoising process that iteratively fuses visible-style infrared and visible features. In this way, the fused image falls into the visible domain and can be directly applied to existing downstream machine systems. Thanks to advancements in image inversion, FusionINV can directly produce fused images in a training-free manner. Extensive experiments demonstrate that FusionINV achieves outstanding performance in both human visual evaluation and machine perception tasks.
---

## 🔧 Virtual Environment Setup

```bash
conda create -n fusioninv python=3.9
conda activate fusioninv
pip install -r requirements.txt
````

---

## 🧪 Testing

1. Download the pretrained [Stable Diffusion (v1.5)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) weights.
2. Config the file `units/model_units.py` or place them in `./pretrained/` folder.
3. Run the following script for IVF testing:

```bash
python fusioninv.py --vis_image_path ./data/in_vis.png --ir_image_path ./data/in_ir.png --output_path ./output --domain_name sky --use_masked_adain False --contrast_strength 1.1 --swap_guidance_scale 1.5 --skip_steps 10 --direction_step_size -0.12 --seed 1 
```

4. Modify the input path in the `fusioninv.py` file to adapt to your test data.

---

## 🚀 Inference Pipeline

* **Step 1**: Invert infrared and visible images into latent noise space using pretrained SD encoder.
* **Step 2**: Use visible noise as style guidance for infrared features.
* **Step 3**: Iteratively denoise using Stable Diffusion and inject fusion-aware features.
* **Step 4**: Decode the final noise into a fused visible-style image.

This process is **training-free**, and fully leverages the generative power of Stable Diffusion.

---

## 📂 Project Structure

```bash
FusionINV/
├── pretrained/                # pretrained diffusion model
├── data/                      # test datasets
├── models/                    # inversion and fusion modules
├── units/                     # utility scripts
├── fusioninv.py               # main test file
└── requirements.txt
```

---

## 📌 Citation

If this work helps your research, please consider citing us:

```bibtex
@article{liang2025fusioninv,
  title={FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion},
  author={Liang, Pengwei and Jiang, Junjun and Ma, Qing and Wang, Chenyang and Liu, Xianming and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```

---

## 🔍 Acknowledgements

Our work builds upon the pretrained [Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) model. We thank the developers for their open-source contributions.



# Instructional Image Editor

**Michigan Data Science Team (MDST)**

Generative AI and latent diffusion models are transforming computer vision from simple classification to complex, creative generation. This project builds an instruction-guided image editing pipeline that starts with mathematical first principles and ends with a robust, interactive generative model.

Our core objective is to replicate the architectural modifications and training pipeline of the *InstructPix2Pix* paper. Rather than generating our own synthetic training data, we will use the authors' official dataset. Our focus is strictly on the model engineering: downloading the pretrained Stable Diffusion backbone, surgically modifying the U-Net architecture, and fine-tuning it to follow natural language image-editing instructions.

By the end of the semester, we will have a fully functional text-to-image editor and will split into specialized research teams to explore cutting-edge ways to improve the base model.

## 🎯 Semester Goals & Roadmap

The project is structured into three distinct phases to take members from beginners to advanced generative AI engineers:

* **Phase 1: Diffusion from Scratch (The MNIST Sandbox)**
We will begin the semester by demystifying the "black box" of diffusion. Members will learn the fundamental mathematics by implementing a basic forward noise scheduler and reverse denoising U-Net from scratch in PyTorch. We will train this lightweight model on the MNIST dataset to generate handwritten digits, ensuring everyone understands the core Ordinary Differential Equations (ODEs) and loss functions that power all diffusion models.
* **Phase 2: The InstructPix2Pix Baseline**
Transitioning from pixel space to latent space, we will replicate the *InstructPix2Pix* training process. Members will download the pretrained weights for Stable Diffusion 1.5 and manually modify its first convolutional layer to accept 8 channels (the noisy latent + the original image latent). We will then write the custom training loop and Dual Classifier-Free Guidance (CFG) sampling logic required to fine-tune the model on the official open-source dataset.
* **Phase 3: Open-Ended Research & Add-ons**
Once the baseline model successfully edits images, the project shifts to open-ended research. Members can stay as one large group or split into specialized teams to "make the model better." A primary direction will be exploring a **Recursive Cross-Turn Attention** mechanism—intercepting the U-Net's Key/Value cache to allow the model to "remember" previous edits, enabling continuous, multi-step image editing without semantic drift or layout hallucination.

## 🧠 What Members Will Learn

This project is highly technical and hands-on. Members will write their own PyTorch implementations rather than relying on high-level APIs.

* **Diffusion Mathematics:** Understand forward Markov noising schedules, variance scaling, and reverse ODE solvers (DDPM and DDIM).
* **Latent Space Engineering:** Work with Variational Autoencoders (VAEs) to compress high-dimensional pixel data into dense perceptual latents.
* **Multimodal Conditioning:** Learn how CLIP text encoders tokenize natural language and how Multi-Head Cross-Attention injects those semantic instructions into spatial image features.
* **PyTorch Model Surgery:** Gain experience hacking into large, pretrained architectures to modify layer dimensions (e.g., expanding Conv2d input channels) while using zero-initialization tricks to maintain training stability.
* **Compute Optimization:** Master standard industry practices for training billion-parameter models on limited hardware, including mixed-precision training (`fp16`/`bf16`), gradient accumulation, and Distributed Data Parallel (DDP) pipelines.
* **Advanced Evaluation:** Move beyond basic accuracy metrics to implement structural and perceptual benchmarking using Directional CLIP similarity and LPIPS.

## 📚 Core References

* **Paper:** *[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)* (Brooks et al., CVPR 2023)
* **Official Repository:** [timothybrooks/instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix)
* **Primary Dataset:** Hugging Face `timbrooks/instructpix2pix-clip-filtered`

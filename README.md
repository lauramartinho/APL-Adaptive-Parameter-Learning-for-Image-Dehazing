## APL: Adaptive Parameter Learning for Image Dehazing paper repository
**This is the source code for the paper "APL: Adaptive Parameter Learning for Image Dehazing" accepted as an oral at IEEE ICMLA (International Conference of Machine Learning and Applications) 2024.**


<p align="center"> 
    <img alt="comparison" width="400" src="https://github.com/user-attachments/assets/6ae3e8e7-419f-4644-a9a0-cc153edfef7e"><br>
</p>

> **Overview of the proposed method. We propose a CNN-based approach to learn and estimate the best parameters for a novel adaptive ISP pipeline in the process of enhancing the quality of hazy images. The proposed regression process acquires knowledge from different fog conditions (like saturation, low contrast, lighting, scattering and blurry in haze), enabling the estimation of optimal parameters for different hazy images. The proposed APL approach presents high accuracy even in different haze conditions, with a light-weight learning process.**

Foggy or hazy images result from light scattering and absorption by atmospheric particles. Intensity transformation techniques offer solutions to solve this problem, but parameters selection significantly impacts the quality of the final image. In this paper we propose the APL method, a parameter learning approach for enhancing hazy images using Convolutional Neural Networks (CNNs), resulting in a new ISP learning-based pipeline. A set of intensity transformation techniques is applied, combined with image quality metrics, to define parameters for hazy image enhancement. A CNN regression model is employed to learn about the problem and estimate parameters for the transformation stage. The best dehazing parameters are determined and utilized to enhance the quality of degraded images. Experiments are conducted on three datasets of hazy images, including two datasets available in the literature and one proposed dataset of real-world foggy images. Results are evaluated comparing to other dehazing methods using full-reference (PSNR and SSIM) and non-reference (NIQE and BRISQUE) metrics, demonstrating a high accuracy in image dehazing achieved by our method.

<p align="center"> 
    <img alt="comparison" width="400" src="https://github.com/user-attachments/assets/acdf3f44-5a88-4544-944a-4fefac655ead"><br>
</p>

> **We also propose the Aerial Hazy Images (AHI) dataset, a challenging dataset composed by over 4700 hazy images, acquired in urban and natural outdoor environments. The proposed dataset comprises hazy images with intense fog and low visibility and scattering. To the best of our knowledge this is the first dataset of hazy images from urban and natural outdoor environments, composed of aerial images.**



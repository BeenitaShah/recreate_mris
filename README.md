<div align="center">

# MRI Reconstruction
</div>

Medical imaging is crucial in modern clinics whose major objective is to acquire high quality medical images for clinical usage at the minimal cost and risk to the patients. (Zhang and Dong, 2019) Accelerating Magnetic Resonance Imaging (MRI) by taking fewer measurements has the potential to reduce medical costs, minimise stress to patients and make MRI possible in applications where it is currently slow or expensive (Zbontar et al. 2018).

This task used a subset of the fastMRI dataset to train a deep learning model to complete the task of MRI reconstruction. This task focused solely on building a model to improve the quality of images generated from under-sampled k-space MRI scans. The solution took an under-sampled K-space volume and output a higher quality resultant image. This problem is a variant to generating higher definition images from low-definition images: The image size cannot be increased and the problem cannot be solved by applying a filter such as a sharpen mask, as all data points in k-space contribute to the image content. Pre and processing and other sanitization techniques have proved useful in improving image quality and model performance in other experiments; however this project does not focus on these aspects.

The dataset used in the project consisted of a training set, 70%, and a test set, 30%. The training set consisted of volumes of K-space images, one volume per patient. A volume is made up of N slices where each slice is a 2D array. Each volume in the training set contained fully sampled K-space data. From this the undersampled K-space data, the ground truth and the under-sampled images were generated.
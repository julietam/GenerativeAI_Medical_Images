# Referencias Bibliográficas

Compilación completa de papers, recursos y herramientas para generación y validación de imágenes médicas.

---

## 📄 Papers Fundamentales

### GANs - Teoría Base

1. **Goodfellow, I., et al. (2014)**  
   "Generative Adversarial Networks"  
   *NeurIPS 2014*  
   [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)  
   🔑 **Paper original de GANs**

2. **Radford, A., Metz, L., & Chintala, S. (2016)**  
   "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"  
   *ICLR 2016*  
   [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)  
   🔑 **DCGAN - Base para medical imaging**

3. **Isola, P., et al. (2017)**  
   "Image-to-Image Translation with Conditional Adversarial Networks"  
   *CVPR 2017*  
   [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)  
   🔑 **Pix2Pix - MRI/CT translation**

4. **Zhu, J.-Y., et al. (2017)**  
   "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"  
   *ICCV 2017*  
   [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)  
   🔑 **CycleGAN - Unpaired translation**

5. **Karras, T., Laine, S., & Aila, T. (2019)**  
   "A Style-Based Generator Architecture for Generative Adversarial Nets"  
   *CVPR 2019*  
   [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)  
   🔑 **StyleGAN - High-resolution synthesis**

---

### Diffusion Models

6. **Ho, J., Jain, A., & Abbeel, P. (2020)**  
   "Denoising Diffusion Probabilistic Models"  
   *NeurIPS 2020*  
   [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)  
   🔑 **DDPM - Teoría base de diffusion**

7. **Rombach, R., et al. (2022)**  
   "High-Resolution Image Synthesis with Latent Diffusion Models"  
   *CVPR 2022*  
   [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)  
   🔑 **Latent Diffusion - Base de Stable Diffusion**

8. **Song, Y., & Ermon, S. (2020)**  
   "Improved Techniques for Training Score-Based Generative Models"  
   *NeurIPS 2020*  
   [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)

---

## 🏥 Medical Imaging - GANs

### Revisiones y Surveys

9. **Yi, X., Walia, E., & Babyn, P. (2019)**  
   "Generative adversarial network in medical imaging: A review"  
   *Medical Image Analysis*  
   [DOI:10.1016/j.media.2019.101552](https://doi.org/10.1016/j.media.2019.101552)

10. **Skandarani, Y., et al. (2021)**  
    "GANs for Medical Image Synthesis: An Empirical Study"  
    *Journal of Medical Imaging*  
    [arXiv:2105.05318](https://arxiv.org/abs/2105.05318)  
    🔍 **Compara DCGAN, LSGAN, WGAN, StyleGAN en medical imaging**

### Brain MRI

11. **Wolterink, J. M., et al. (2017)**  
    "Deep MR to CT Synthesis using Unpaired Data"  
    *SASHIMI Workshop, MICCAI 2017*  
    [arXiv:1708.01155](https://arxiv.org/abs/1708.01155)  
    🧠 **MRI → CT con CycleGAN**

12. **Dar, S. U. H., et al. (2019)**  
    "Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks"  
    *IEEE Transactions on Medical Imaging*  
    [DOI:10.1109/TMI.2019.2901750](https://doi.org/10.1109/TMI.2019.2901750)  
    🧠 **T1 → T2, FLAIR synthesis**

### CT Imaging

13. **Yang, Q., et al. (2018)**  
    "Low-Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss"  
    *IEEE Transactions on Medical Imaging*  
    [DOI:10.1109/TMI.2018.2827462](https://doi.org/10.1109/TMI.2018.2827462)  
    🔬 **Low-dose CT denoising**

14. **Zhang, Y., Yu, H., & Wang, G. (2018)**  
    "A CT Reconstruction Technique Using Anchor-Based Tuning"  
    *Medical Physics*  
    💉 **Artifact reduction en CT**

### Chest X-Ray

15. **Srivastava, A., et al. (2017)**  
    "Lung Segmentation from Chest X-rays with Deep Learning"  
    *SPIE Medical Imaging*  
    📸 **X-ray synthesis para segmentación**

16. **Madani, A., et al. (2018)**  
    "Semi-supervised learning with generative adversarial networks for chest X-ray classification"  
    *Medical Image Analysis*  
    [DOI:10.1016/j.media.2018.08.001](https://doi.org/10.1016/j.media.2018.08.001)

### Histopatología

17. **Hou, L., et al. (2019)**  
    "Unsupervised Histopathology Image Synthesis"  
    *arXiv*  
    [arXiv:1712.05021](https://arxiv.org/abs/1712.05021)  
    🔬 **StyleGAN para tissue synthesis**

### Mamografía

18. **Korkinof, D., et al. (2018)**  
    "High-Resolution Mammogram Synthesis using Progressive GANs"  
    *MICCAI 2018*  
    🎗️ **Progressive GAN para mamografías 1280×1024**

---

## 🌟 Medical Imaging - Diffusion Models

19. **Friedrich, L., et al. (2023)**  
    "Medfusion: A multimodal comparison of DDPMs and GANs for medical image synthesis"  
    *Scientific Reports*  
    [DOI:10.1038/s41598-023-34341-2](https://doi.org/10.1038/s41598-023-34341-2)  
    🔑 **Estudio comparativo: Diffusion supera GANs en diversidad**  
    📊 **Recall: 0.40 (Diffusion) vs 0.19 (GANs)**

20. **Wu, J., et al. (2023)**  
    "MedSegDiff: Medical Image Segmentation with Diffusion Models"  
    *MICCAI 2023*  
    [arXiv:2301.11798](https://arxiv.org/abs/2301.11798)

21. **Kazerouni, A., et al. (2023)**  
    "Diffusion Models in Medical Imaging: A Comprehensive Survey"  
    *Medical Image Analysis*  
    [arXiv:2211.07804](https://arxiv.org/abs/2211.07804)  
    📚 **Survey exhaustivo de diffusion en medicina**

---

## 📊 Revisiones Recientes (2024-2025)

22. **Oulmalme, L., et al. (2025)**  
    "A systematic review of generative AI approaches for medical image enhancement"  
    *Medical Image Analysis (In Press)*  
    🆕 **Compara GANs, Transformers, Diffusion - Abril 2025**

23. **Ibrahim, H., et al. (2025)**  
    "Generative AI for synthetic data generation across multiple medical modalities: a comprehensive review"  
    *Computers in Biology and Medicine - Marzo 2025*  
    🆕 **Multimodal synthesis review**

24. **Fahad, S., et al. (2025)**  
    "Developments in deep learning approaches for medical image analysis and diagnosis"  
    *PMC - Mayo 2025*  
    [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11077369/)

---

## 🔬 Métricas y Evaluación

25. **Heusel, M., et al. (2017)**  
    "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"  
    *NeurIPS 2017*  
    [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)  
    📏 **FID (Fréchet Inception Distance)**

26. **Salimans, T., et al. (2016)**  
    "Improved Techniques for Training GANs"  
    *NeurIPS 2016*  
    [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)  
    📏 **Inception Score (IS)**

27. **Sajjadi, M. S. M., et al. (2018)**  
    "Assessing Generative Models via Precision and Recall"  
    *NeurIPS 2018*  
    [arXiv:1806.00035](https://arxiv.org/abs/1806.00035)  
    📏 **Precision-Recall para GANs**

---

## 🏗️ Frameworks y Herramientas

### MONAI

28. **Cardoso, M. J., et al. (2022)**  
    "MONAI: An open-source framework for deep learning in healthcare"  
    *arXiv*  
    [arXiv:2211.02701](https://arxiv.org/abs/2211.02701)  
    🛠️ **Framework oficial de MONAI**

**Website**: [https://monai.io/](https://monai.io/)  
**GitHub**: [https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

### medigan

29. **Osuala, R., et al. (2023)**  
    "medigan: a Python library of pretrained generative models for medical image synthesis"  
    *Journal of Medical Imaging*  
    [DOI:10.1117/1.JMI.10.6.061403](https://doi.org/10.1117/1.JMI.10.6.061403)  
    🛠️ **21+ modelos pre-entrenados**

**GitHub**: [https://github.com/RichardObi/medigan](https://github.com/RichardObi/medigan)

---

## 🗄️ Datasets Públicos

### Brain MRI

30. **IXI Dataset**  
    *Imperial College London*  
    URL: [https://brain-development.org/ixi-dataset/](https://brain-development.org/ixi-dataset/)  
    🧠 **600 MRI scans: T1, T2, PD, MRA, DWI**

31. **BraTS (Brain Tumor Segmentation)**  
    *MICCAI Challenge*  
    URL: [https://www.med.upenn.edu/cbica/brats/](https://www.med.upenn.edu/cbica/brats/)  
    🧠 **~2000 brain MRI con segmentación de tumores**

### CT & X-Ray

32. **ChestX-ray14**  
    Wang, X., et al. (2017). "ChestX-ray8: Hospital-scale chest X-ray database"  
    *CVPR 2017*  
    URL: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)  
    📸 **112,120 chest X-rays de 30,805 pacientes**

33. **CheXpert**  
    Irvin, J., et al. (2019). "CheXpert: A large chest radiograph dataset"  
    *AAAI 2019*  
    URL: [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)  
    📸 **224,316 chest radiographs**

### MRI Reconstruction

34. **fastMRI**  
    Zbontar, J., et al. (2018). "fastMRI: An Open Dataset and Benchmarks for MRI Reconstruction"  
    *arXiv:1811.08839*  
    URL: [https://fastmri.org/](https://fastmri.org/)  
    🧲 **~1.5M knee & brain MRI raw k-space data**

### Multi-modal

35. **Medical Segmentation Decathlon**  
    URL: [http://medicaldecathlon.com/](http://medicaldecathlon.com/)  
    🎯 **10 datasets: CT, MRI (brain, liver, lung, prostate, etc.)**

36. **The Cancer Imaging Archive (TCIA)**  
    URL: [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/)  
    🎗️ **Millones de imágenes médicas oncológicas**

---

## 🎓 Tutoriales y Cursos

37. **MONAI Tutorials**  
    GitHub: [https://github.com/Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)  
    📚 **50+ notebooks para medical imaging**

38. **PyTorch GAN Tutorial**  
    URL: [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

39. **fastMRI Tutorial**  
    GitHub: [https://github.com/facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI)

---

## 📖 Libros

40. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**  
    *Deep Learning*  
    MIT Press  
    URL: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)  
    📘 **Libro fundamental de deep learning**

41. **Prince, S. J. D. (2023)**  
    *Understanding Deep Learning*  
    MIT Press  
    URL: [https://udlbook.github.io/udlbook/](https://udlbook.github.io/udlbook/)  
    📘 **Incluye capítulo sobre GANs y Diffusion**

---

## 🔐 Ética y Privacidad

42. **Abadi, M., et al. (2016)**  
    "Deep Learning with Differential Privacy"  
    *ACM CCS 2016*  
    [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)  
    🔒 **Differential privacy en deep learning**

43. **Chen, R. J., et al. (2021)**  
    "Synthetic data in machine learning for medicine and healthcare"  
    *Nature Biomedical Engineering*  
    [DOI:10.1038/s41551-021-00751-8](https://doi.org/10.1038/s41551-021-00751-8)  
    🔒 **Consideraciones éticas de datos sintéticos**

---

## 🌐 Comunidades y Recursos Online

- **Papers with Code - Medical Imaging**  
  [https://paperswithcode.com/area/medical](https://paperswithcode.com/area/medical)

- **MICCAI (Medical Image Computing)**  
  [https://www.miccai.org/](https://www.miccai.org/)

- **Grand Challenges**  
  [https://grand-challenge.org/](https://grand-challenge.org/)

- **MONAI Slack**  
  [https://projectmonai.slack.com](https://projectmonai.slack.com)

- **r/MachineLearning (Reddit)**  
  [https://www.reddit.com/r/MachineLearning/](https://www.reddit.com/r/MachineLearning/)

---

## 📰 Conferencias Relevantes

- **MICCAI** (Medical Image Computing and Computer Assisted Intervention)
- **CVPR** (Computer Vision and Pattern Recognition)
- **NeurIPS** (Neural Information Processing Systems)
- **ICLR** (International Conference on Learning Representations)
- **MIDL** (Medical Imaging with Deep Learning)
- **ISBI** (IEEE International Symposium on Biomedical Imaging)

---

## 🔄 Actualizaciones

Este documento se actualiza regularmente con nuevos papers y recursos.

**Última actualización**: Noviembre 2025

---

## 📝 Cómo Citar

Si usas este repositorio en tu investigación, por favor cita:

```bibtex
@misc{medicalimagegeneration2025,
  title={Generación y Validación de Imágenes Médicas: Retos, Riesgos y Oportunidades},
  author={[Tu Nombre]},
  year={2025},
  publisher={GitHub},
  url={https://github.com/tu-usuario/medical-image-generation}
}
```

---

**Contribuciones**: Si conoces papers relevantes que faltan, por favor abre un PR o issue.
# Reducing annotation burden in MR: A novel MR-contrast guided contrastive learning approach for image segmentation

The code associated with the manuscript [Reducing annotation burden in MR: A novel MR-contrast guided contrastive learning approach for image segmentation](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16820) is available here.

### Abstract:
The availability of limited labeled data for supervised medical image segmentation tasks motivates the use of representational learning techniques for deep learning (DL) models. The current work presents a novel MR contrast guided self-supervised contrastive learning approach to reduce the annotation burden in MR segmentation tasks. The proposed contrastive learning strategy shows a) the feasibility to learn tissue specific information that controls MR image contrast using information from a large set of unlabeled multi-contrast data, and b) its application to improve performance on subsequent segmentation tasks on single contrast data. The results, across multiple datasets and segmentation tasks, demonstrate that learning to embed underlying tissue-specific information can leverage the limited availability of labeled data and, in turn, reduce the associated annotation burden on clinicians.



![Figure1](https://github.com/lunastra26/multi-contrast-contrastive-learning/assets/60745251/400e6145-8ece-4ad8-9af0-8848a63005c5)

 
### Offline constraint map generation
***
1) Identify an appropriate multi-contrast space for your downstream segmentation task: **run generate_constraint_maps.py**

Example: For segmentation tasks in T2-weighted images, a set of co-registered MR images where MR contrast varies depending on underlying T2 such as multiple echo images with varying T2-weightings can be used to learn T2 information

Other examples with respect to Brain Tumor Segmentation (BraTS) dataset:
- For segmentation tasks in T1-weighted images, generate constraints maps from T1 contrast images. e.g., [T1Gd, T1w]
- For segmentation tasks in T2-weighted images, generate constraints maps from T2 contrast images. e.g., [T2w, T2-FLAIR]
- For segmentation tasks in T1-weighted and T2-weighted images, generate constraints maps from T1 and T2 contrast images. e.g., [T1Gd, T1w, T2w, T2-FLAIR]

2) Generate training data with image and corresponding constraint maps: run **generate_h5_pretraining.py**

### Constrained Contrastive Learning
***
Pretrainining the DL model to embed MR constrast information: run **constrained_contrastive_learning.py**
- For segmentation tasks on anatomical regions i.e., regions with fixed spatial locations in the body such as liver/spleen, it is recommended to use patch size of 4x4 and warm start. The encoder can be pretrained with global contrastive learning (optional with pretrain_global_contrastive_learning.py), and the full decoder can be pretrained with constrained contrastive learning.

 - For segmentation tasks such as tumors or lesion detection that have no fixed spatial location in the body, it is recommended to partially train decoder with constrained contrastive learning  

The tissue-specific representations learned by the pretraining process can be visualized by extracting feature maps from layers before projection head.

### Finetune for downstream task
***
1) For downstream segmentation tasks, use the pretrained model with a loss function of choice


If you find this work useful, please consider citing the following:
Umapathy, L, Brown, T, Mushtaq, R, Greenhill M, Lu J, Martin D, Altbach M, and Bilgin A. Reducing annotation burden in MR: A novel MR-contrast guided contrastive learning approach for image segmentation. Med Phys. 2023; 1-14. https://doi.org/10.1002/mp.16820




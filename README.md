# Reducing annotation burden in MR: A novel MR-contrast guided contrastive learning approach for image segmentation


The availability of limited labeled data for supervised medical image segmentation tasks motivates the use of representational learning techniques for deep learning (DL) models. The current work presents a novel MR contrast guided self-supervised contrastive learning approach to reduce the annotation burden in MR segmentation tasks. The proposed contrastive learning strategy shows a) the feasibility to learn tissue specific information that controls MR image contrast using information from a large set of unlabeled multi-contrast data, and b) its application to improve performance on subsequent segmentation tasks on single contrast data. The results, across multiple datasets and segmentation tasks, demonstrate that learning to embed underlying tissue-specific information can leverage the limited availability of labeled data and, in turn, reduce the associated annotation burden on clinicians.


This work can be cited as:
Umapathy L, Brown T, M.B., Mushtaq R, Greenhill M, Lu J, Martin D, Altbach M, and Bilgin A. Reducing annotation burden in MR: A novel MR-contrast guided contrastive learning approach for image segmentation.


Steps to train
## Offline constraint map generation
1) Identify an appropriate multi-contrast space for your downstream segmentation task. 
Example: For segmentation tasks in T2-weighted images, use a set of co-registered MR images where MR contrast varies depending on underlying T2 such as multiple echo images with varying T2-weightings
With respect to BraTS dataset:
a) For segmentation tasks in T1-weighted images, generate constraints maps from T1 contrast images. e.g., [T1Gd, T1w]
b) For segmentation tasks in T2-weighted images, generate constraints maps from T2 contrast images. e.g., [T2w, T2-FLAIR]
c) For segmentation tasks in T1-weighted and T2-weighted images, generate constraints maps from T1 and T2 contrast images. e.g., [T1Gd, T1w, T2w, T2-FLAIR]

2) Generate HDF5 files for image and corresponding constraint maps

## Constrained Contrastive Learning
Run constrained_contrastive_learning.py for pretrainining the DL model to embed MR constrast information
1) For segmentation tasks on anatomical regions i.e., regions with fixed spatial locations in the body such as liver/spleen, 
a) Recommended to use a Full decoder
b) Recommended to use patch size of 4x4
c) Recommended to use warm start: pretrain Encoder with global contrastive learning, decoder with local contrastive learning for best results

1) For segmentation tasks on abnormal regions such as tumors or lesions that have no fixed spatial location in the body
a) Recommended to use a partial decoder

## Finetune for downstream task
1) For downstream segmentation tasks, use the pretrained model
### Remarks: The associated code will be released on acceptance of the manuscript



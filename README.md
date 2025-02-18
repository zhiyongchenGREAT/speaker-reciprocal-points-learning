# Efficient Open-Set Speaker Identification through Few-shot Tuning with Speaker Reciprocal Points and Unknown Samples\

## Overview
Speaker recognition systems face significant challenges due to intrinsic speaker variability, such as aging and emotional fluctuations, which can undermine accuracy. Traditional solutions often rely on large-scale pretraining or repeated enrollments, which are impractical for real-world applications. In this paper, we propose a novel framework to address these challenges by leveraging speaker-specific zero-shot text-to-speech (TTS) systems. Our approach generates synthetic speech that captures age-related and emotional variations in voice, followed by few-shot open-set fine-tuning on limited real enrollment data. We demonstrate that this strategy effectively enhances robustness to intrinsic variability, requiring minimal enrollment samples. Our method is evaluated on multiple challenging datasets, showing its strong generalization ability for long-term, time-varying, and emotionally rich speaker recognition tasks.

<p align="center">
  <img src="images/srpl_arch.png" alt="SRPL+ Architecture" width="50%" />
</p>
<!-- <p align="center">
  <img src="images/srpl.png" alt="SRPL+ Process" width="50%" />
</p> -->

## Dataset
We utilize two primary datasets in our research:

**Qualcomm Speech**: Dataset links and our experimental settings.

[Link to Qualcomm Speech dataset](https://developer.qualcomm.com/project/keyword-speech-dataset)

**FFSVC HiMia**: Dataset links and our experimental settings.

[Link to HiMia dataset](https://aishelltech.com/wakeup_data)

**Split for training and testing**:

[Example split](https://github.com/srplplus/srplplus.github.io/tree/main/QSpeech_wavLMTDNN_embs/emb_test)

## Pretrained Audio Foundation Models
Our method is built upon a pretrained audio foundation model EResNetV2 and WavLM-base-plus.

[Link and Details to the pretrained WavLM-TDNN](https://huggingface.co/microsoft/wavlm-base-plus-sv)

<!-- <p align="center">
  <img src="images/wavlm.png" alt="SRPL+ Architecture" width="50%" />
</p> -->

## Evaluations
The evaluation section details the performance metrics on open-set speaker identification. The Open Set Classification Rate (OSCR) calculates the area under the curve mapping the Correct Classification Rate (CCR) for known classes to the False Positive Rate (FPR) for unknown data, offering a threshold-independent evaluation for open-set.

<!-- $CCR(TH) = \frac{|\{x \in TestData^{k} \mid \arg\max_{k} P(k|x) = k \cap P(k|x) \geq TH\}|}{|TestData^{k}|}$

$FPR(TH) = \frac{|\{x \mid x \in Unknown \cap \max_k P(k|x) \geq TH\}|}{|Unknown|}$ -->
![emb plot](images/eq.png)

We provide the implementation of core evaluation metrics, along with other evaluation metrics, in our code repository. An inference script is also provided to evaluate the model on our example testing split data.

[Inference script and evaluation metrics implementation](https://github.com/srplplus/srplplus.github.io/blob/main/inference_demo.ipynb)

## Code
Code used in this research for model training, and evaluation, is available for public use after publication. This encourages reproducibility and further experimentation in the field.

[SRPL+ code repository](https://github.com/srplplus/srplplus.github.io)

<!-- ## Visualization and Evaluations
We present a series of visualizations and detailed evaluations to illustrate our method's effectiveness as in the paper. The t-sne embedding plots clearly demostrate the effectiveness of our method. -->

<!-- ![emb plot](images/emb_srpl.png) -->

<!-- [Link to visualizations and detailed evaluations]() -->

<!-- ## How to Use
This section provides a step-by-step guide on how to replicate our research findings, including setting up the environment, preprocessing the data, training the model, and conducting evaluations. -->

## Citation
Please cite our work if it contributes to your research:

@inproceedings{chen2024enhancing,
  title={Enhancing Open-Set Speaker Identification Through Rapid Tuning With Speaker Reciprocal Points and Negative Sample},
  author={Chen, Zhiyong and Ai, Zhiqi and Li, Xinnuo and Xu, Shugong},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  pages={1144--1149},
  year={2024},
  organization={IEEE}
}

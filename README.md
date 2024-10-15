# Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Sample.(IEEE SLT24)

## Overview
We introduces a novel framework for open-set speaker identification in household environments, playing a crucial role in facilitating seamless human-computer interactions. Addressing the limitations of current speaker models and classification approaches, our work integrates an AudioLM frontend with a few-shot rapid tuning neural network (NN) backend for enrollment, employing task-optimized Speaker Reciprocal Points Learning (SRPL) to enhance discrimination across multiple target speakers. Furthermore, we propose an enhanced version of SRPL (SRPL+), which incorporates negative sample learning with both speech-synthesized and real negative samples to significantly improve open-set SID accuracy. Our approach is thoroughly evaluated across various multi-language datasets, demonstrating its effectiveness in achieving high usability for complex household multi-speaker scenarios and significantly enhancing open-set performance.

<p align="center">
  <img src="images/srpl_arch.png" alt="SRPL+ Architecture" width="50%" />
</p>
<p align="center">
  <img src="images/srpl.png" alt="SRPL+ Process" width="50%" />
</p>

## Dataset
We utilize two primary datasets in our research:

**Qualcomm Speech**: Dataset links and our experimental settings.

[Link to Qualcomm Speech dataset](https://developer.qualcomm.com/project/keyword-speech-dataset)

| Target Speakers | Unknown Speakers | N-fold |
|----------|----------|----------|
| 5/10 | 10 | 3 |

**FFSVC HiMia**: Dataset links and our experimental settings.

[Link to HiMia dataset](https://aishelltech.com/wakeup_data)

| Target Speakers | Unknown Speakers | N-fold |
|----------|----------|----------|
| 5/10 | 10 | 3 |

**Example split for training and testing**:

[Example split](https://github.com/srplplus/srplplus.github.io/tree/main/QSpeech_wavLMTDNN_embs/emb_test)

## Pretrained Audio Large Model
Our methodology is built upon a pretrained audio large model WavLM-base-plus for TDNN speaker verification, specifically designed to capture the nuances of human speech and speaker characteristics. This model serves as the foundation for our rapid tuning process, allowing for effective speaker identification. We use the 512 dimensional WavLM-base-plus with TDNN extracted speaker embedding for our backend rapid tuning and enrollment (SRPL+) models.

[Link and Details to the pretrained WavLM-TDNN AudioLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)

<p align="center">
  <img src="images/wavlm.png" alt="SRPL+ Architecture" width="50%" />
</p>

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

## Visualization and Evaluations
We present a series of visualizations and detailed evaluations to illustrate our method's effectiveness as in the paper. The t-sne embedding plots clearly demostrate the effectiveness of our method.

![emb plot](images/emb_srpl.png)

<!-- [Link to visualizations and detailed evaluations]() -->

<!-- ## How to Use
This section provides a step-by-step guide on how to replicate our research findings, including setting up the environment, preprocessing the data, training the model, and conducting evaluations. -->

## Citation
Please cite our work if it contributes to your research:

-Chen, Zhiyong, et al. "Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Sample." arXiv preprint arXiv:2409.15742 (2024).

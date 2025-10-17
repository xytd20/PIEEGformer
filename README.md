# PIEEGformer
This repository provides the official PyTorch implementation for the **PI-EEGformer**, a model designed for predicting movement transitions from EEG signals, as detailed in the paper "Physiology-Inspired EEG Transformer for Predicting Movement Transitions in Bimanual Tasks."

* **PI-EEGformer was tested on the following dataset and achieved SOTA results**:
    1. A **private dataset** from a bimanual Go/No-go experiment designed to simulate sudden movement changes during an ongoing motor task.

## Tasks Supported

The codebase allows for training and evaluation on the following classification task:

* **On the Bimanual Movement Dataset:**
    ***Movement Transition Prediction:** Binary classification (Go vs. No-go) to predict an impending unilateral wrist extension during a steady bimanual contraction[cite: 62, 282]. [cite_start]The prediction uses EEG data from a 400 ms window (500 ms to 100 ms) before the actual movement begins[cite: 353].

## Related Paper

The model and experiments are described in our paper:
**Physiology-Inspired EEG Transformer for Predicting Movement Transitions in Bimanual Tasks**

> **Abstract:** Human-machine interfaces (HMIs) are widely used in motor rehabilitation and augmentation. Forecasting movement transitions is crucial for system safety and reactivity, especially when anticipating human motor intentions in response to sudden perturbations. In this study, we investigated pre-movement neural signatures preceding sudden movement transitions during ongoing bimanual tasks. Informed by these findings, we propose a physiology-informed EEG Transformer (PI-EEGformer). An EEG dataset was collected from a bimanual task where one hand had to switch motor states in response to unexpected cues. The PI-EEGformer achieved an average accuracy of 0.912 in inter-subject tests and 0.829 in cross-subject tests, outperforming seven state-of-the-art models. These results demonstrate that EEG neural signatures can predict sudden movement transitions and that the PI-EEGformer enables accurate predictions, contributing to more responsive and realistic HMI systems.

## Contact

For questions about the paper or code, please use GitHub Issues or contact my email: hp375169@gmail.com, or contact the corresponding author mentioned in our paper:
* [cite_start]**Ciaran McGeady:** `c.mcgeady@imperial.ac.uk`
* [cite_start]**Chong Li:** `chongli@tsinghua.edu.cn`
* [cite_start]**Dario Farina:** `d.farina@imperial.ac.uk`

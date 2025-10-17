# PIEEGformer
This repository provides the official PyTorch implementation for the **PI-EEGformer**, a model designed for predicting movement transitions from EEG signals, as detailed in the paper "Physiology-Inspired EEG Transformer for Predicting Movement Transitions in Bimanual Tasks."

* **PI-EEGformer was tested on the following dataset and achieved SOTA results**:
    1. A **private dataset** from a bimanual Go/No-go experiment designed to simulate sudden movement changes during an ongoing motor task.

## Tasks Supported

The codebase allows for training and evaluation on the following classification task:

* **On the Bimanual Movement Dataset:**
    **Movement Transition Prediction:** Binary classification (Go vs. No-go) to predict an impending unilateral wrist extension during a steady bimanual contraction. The prediction uses EEG data from a 400 ms window (500 ms to 100 ms) before the actual movement begins.

## Related Paper

The model and experiments are described in our paper:
**Physiology-Inspired EEG Transformer for Predicting Movement Transitions in Bimanual Tasks**
Which was accepted by IEEE JBHI on October 14th.

## Contact

For questions about the paper or code, please use GitHub Issues or contact my email: hp375169@gmail.com, or contact the corresponding author mentioned in our paper:
* **Ciaran McGeady:** `c.mcgeady@imperial.ac.uk`
* **Chong Li:** `chongli@tsinghua.edu.cn`
* **Dario Farina:** `d.farina@imperial.ac.uk`

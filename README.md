# DART: Disentanglement of Accent and Speaker Representation in Multispeaker TTS

This repository contains the official demo code for **DART**, accepted at the *Audio Imagination Workshop, NeurIPS 2024*.

[Audio samples](https://amaai-lab.github.io/DART/)

[Paper](https://arxiv.org/abs/2410.13342)


## Overview

DART disentangles speaker identity and accent representation in multispeaker TTS using a structured latent framework.



## Training

Train on L2-ARCTIC:

```CUDA_VISIBLE_DEVICES=0 python train.py --dataset L2ARCTIC```

---

## Inference

Two synthesis scripts are provided:

- `synthesize_converted.py `
  Generates speech across combinations of speakers, accents, and sentences.

- `synthesize_stats_valset.py`  
  Generates speech from a `metadata .txt` file.

### Required preprocessing

Before inference, extract embeddings:

```python extract_stats.py```

This saves MLVAE-based embeddings for speakers and accents.



### Example

```CUDA_VISIBLE_DEVICES=0 python synthesize_converted.py   --dataset L2ARCTIC   --restore_step 704000```


## Citation

If you find this model useful, please cite our paper: 
```
@inproceedings{melechovsky2024dart,
  title={DART: Disentanglement of Accent and Speaker Representation in Multispeaker Text-to-Speech},
  author={Melechovsky, J. and Mehrish, A. and Sisman, B. and Herremans, D.},
  booktitle={Audio Imagination Workshop, NeurIPS},
  year={2024}
}
```


## Acknowledgements

Based on Comprehensive Transformer TTS by Keon Lee et al.

## Contact

Open an issue for questions or collaboration.

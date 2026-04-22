# DART: Disentanglement of Accent and Speaker Representation in Multispeaker TTS
This repository contains the official demo code for **DART**, accepted at the *Audio Imagination Workshop, NeurIPS 2024*.
- 🎧 **Audio samples**: https://amaai-lab.github.io/DART/  
- 📄 **Paper codebase**: This repository  
The implementation builds on the excellent  
[Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS).
---
## 🚀 Overview
DART focuses on disentangling **speaker identity** and **accent representation** in multispeaker text-to-speech systems using a structured latent approach.
---
## 📦 Installation
*(Add setup instructions here if needed.)*
---
## 🏋️ Training
To train DART on the L2-ARCTIC dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset L2ARCTIC

⸻

🔊 Inference

Two synthesis scripts are provided:

* synthesize_converted.py
    Generates speech from predefined scripts across combinations of speakers, accents, and sentences.
* synthesize_stats_valset.py
    Generates speech from a metadata .txt file.

⚠️ Important

Before running any synthesis script, you must first extract latent embeddings:

python extract_stats.py

This step computes and stores the MLVAE-based embeddings for speakers and accents, which are required for inference.

Example

CUDA_VISIBLE_DEVICES=0 python synthesize_converted.py \
    --dataset L2ARCTIC \
    --restore_step 704000

⸻

📖 Citation

If you find this work useful, please cite:

@inproceedings{melechovsky2024dart,
  title={DART: Disentanglement of Accent and Speaker Representation in Multispeaker Text-to-Speech},
  author={Melechovsky, J. and Mehrish, A. and Sisman, B. and Herremans, D.},
  booktitle={Audio Imagination Workshop, NeurIPS},
  year={2024}
}

⸻

🙏 Acknowledgements

This work builds upon:

* Comprehensive Transformer TTS by Keon Lee et al.

⸻

📬 Contact

For questions or collaborations, feel free to open an issue.

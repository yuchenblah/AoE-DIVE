<div align="center">
  <h1>An Attempt at Router-Free DIVE</h1>
  <div>
    <a href="#overview">📝 Overview</a> | <a href="#installation">⚙️ Installation Guide</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#evaluation">💎 Evaluation</a>
  </div>
</div>

<h2 id="overview">📝 Overview</h2>

<h2 id="installation">⚙️ Installation</h2>

Step 1: Create a new conda environment:
```
conda create -n aoe_dive python=3.9
conda activate aoe_dive
```
Step 2: Install relevant packages
```
conda install pytorch==2.7.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<h2 id="quick-start">🚀 Quick Start</h2>

Modeling file in ./models/modeling_aoe_dive.py

Take **DIVE 1/8** as an example.

---

### **Step 1: Domain Affinity Mining & Pruning**

```bash
bash scripts/prune/prune_tinyllama_mlp_0.5.sh
```

### **Step 2: Reconstruct the DIVE model**

```bash
bash scripts/establish/establish_tinyllama_8_1_0.5.sh
```

* Uses: `configurations/tinyllama_smoe/config_8_1_0.5.json`.

<h2 id="evaluation">💎 Evaluation</h2>

In ./scripts/aoe_dive_test

```bash
bash scripts/aoe_dive_test/ppl.sh
conda activate lmeh
bash scripts/aoe_dive_test/lmeh_moe.sh
bash scripts/aoe_dive_test/lmeh_moe_fewshot.sh
```

<h2 id="citation">💬 Citation</h2>


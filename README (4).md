# LLM-sudoku-solver  
Teach a small language model to crack 4×4 Sudoku puzzles with chain-of-thought reasoning.

---

## 1. Abstract  
Solving 4×4 Sudoku is a compact but non-trivial reasoning task: the model must respect row, column, and 2×2 box constraints while filling 3-12 blanks.  
We created a synthetic dataset of 2 k puzzles via the `reasoning-gym` library, fine-tuned **Gemma-3-4B-it** with **GRPO** (Group Relative Policy Optimisation) using four dense rewards, and distilled a portable 4-bit checkpoint that reaches &gt; 96 % exact-match accuracy on a held-out validation set.  
The whole pipeline (data → train → 4-bit export) runs in &lt; 2 h on a single RTX-4090.

---

## 2. Dataset  
| Property        | Value |
|----------------|-------|
| Type           | Mini-Sudoku (4×4) |
| Empty cells    | 3 – 12 per puzzle |
| Train / Val    | 1 800 / 200 |
| Format         | Plain-text grid with `_` for blanks |
| Source         | from huggingface |

---

### 3. Fine-Tuning Approach

We treat **Sudoku** as a *reasoning* rather than a *generation* task and therefore adopt **GRPO**, a variant of PPO that does **not** need a separate value model.

#### 3.1 Model

- **Base:** `google/gemma-3-4b-it` (bf16)
- **Context window:** 1,024 tokens
- **Mode:** Full fine-tune (no LoRA) → best reasoning quality

---

Markdown
Copy
Code
Preview
### 4. Quick Start

#### 4.1 Install
```bash
git clone https://github.com/<user>/mini-sudoku-solver.git
cd mini-sudoku-solver
pip install -r requirements.txt   # unsloth, transformers, trl, datasets, wandb, tqdm
```

#### 4.2 Generate Your Own Dataset
``` bash
python dataset.py   # saves train.json / val.json under ./data
```
#### 4.3 Train
``` bash
python finetune.py  # prompts for optional wandb key
```

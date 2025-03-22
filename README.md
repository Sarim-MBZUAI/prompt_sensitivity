


# When Prompts Vary but Intent Remains: Enforcing Robustness in Vision Language Models

This repository contains the implementation and experimental framework for our research on reducing prompt sensitivity in Vision-Language Models (VLMs). Our work focuses on mitigating hallucinations and ensuring stable, semantically consistent outputs—especially in high-stakes applications like medical imaging and diagnosis.

## Overview

Vision-Language Models (VLMs) have become integral to multimodal tasks such as image captioning and visual question answering. However, even state-of-the-art models can be sensitive to minor prompt variations (e.g., spelling errors, paraphrasing), which can lead to inconsistent and sometimes erroneous outputs. This project proposes a novel training approach that incorporates a **POSIX Loss** function to enforce semantic consistency and context grounding across different prompt variations.

## Key Features

**Novel POSIX Loss Function**  
Penalizes inconsistencies in output probability distributions for semantically equivalent prompts.  

$$
L_{\text{POSIX}} = \frac{1}{N(N-1)} 
\sum_{i=1}^N \sum_{j=1}^N 
\frac{1}{L_{y_j}} 
\log\!\biggl(\frac{P_{\mathcal{M}}(y_j \mid x_i)}{P_{\mathcal{M}}(y_j \mid x_j)}\biggr)
$$

$$
L_{\text{total}} = 100 \cdot L_{\text{POSIX}}
$$


- **Robust Fine-Tuning:**  
  Fine-tunes Medical Large Vision-Language Models (Med-LVLMs) like LLaVA-Med using tailored data augmentations. These augmentations include:
  - Keyboard-aware spelling errors
  - Structural template variations
  - Clinically validated paraphrases

- **Parameter-Efficient Adaptation:**  
  Utilizes Low-Rank Adaptation (LoRA) on key projection matrices while freezing the vision encoder to preserve radiological feature extraction.



## Methodology

Our methodology involves:

1. **Benchmarking VLMs:**  
   We evaluated specialized Med-LVLMs (LLaVA-Med, Med-Flamingo) and general-purpose models (LLaVA 1.5, Qwen-VL, OpenFlamingo) across multiple prompt variations.

2. **Generating Prompt Variations:**  
   We create three systematic prompt variation methods:
   - **Keyboard-aware spelling errors:** Generated using QWERTY adjacency patterns.
   - **Structural template variations:** 20 distinct formats with variations in delimiters and capitalization.
   - **Clinically validated paraphrases:** Produced using the following paraphrase generation prompt:

   ```python
   You are an expert radiologist. Generate exactly 10 different ways to ask this radiology question. Each variation must:
   1. Keep the exact same medical meaning but use different phrasing and terminology
   2. Be a complete, standalone question
   3. Use proper medical terminology appropriate for a radiology report
   4. Each variation must be a proper question about the specific condition mentioned
   5. Be concise and clear

   Original question: "{question}"

   Important: Make each variations unique and meaningful while preserving the exact medical meaning.

   Provide only the variations, one per line, without any additional text or numbering.
   ```

3. **POSIX Loss Computation:**  
   The loss is computed by comparing the length-normalized generation probabilities of responses across different prompt pairs. This enforces consistency, reducing sensitivity to prompt variations.

## Experimental Results

### Zero-Shot Performance

The table below summarizes the zero-shot performance of various models across different prompt types:

| Model         | Template Acc | Template Posix       | Spelling Acc | Spelling Posix      | Paraphrase Acc | Paraphrase Posix      | Model Avg Acc | Model Avg Posix  |
|---------------|--------------|----------------------|--------------|---------------------|----------------|-----------------------|---------------|------------------|
| LLAVA-med     | **0.755**    | **0.0208±0.046**     | **0.716**    | 0.3069±0.17         | **0.619**     | 1.0163±0.216          | **0.696**    | 0.448            |
| Med-flamingo  | 0.207        | 0.3800±0.04          | 0.203        | **0.1590±0.139**    | 0.200          | 0.4240±0.097          | 0.203        | **0.321**        |
| Qwen-VL-Chat  | 0.242        | 1.0070±1.09          | 0.259        | 0.2400±0.47         | 0.245          | 0.5140±0.962          | 0.248        | 0.587            |
| Open-flamingo | 0.204        | 0.3100±0.031         | 0.207        | 0.3360±0.038        | 0.205          | **0.3800±0.070**      | 0.205        | 0.342            |
| LLAVA-1.6     | 0.475        | 2.3919±1.3464        | 0.651        | 1.5800±1.0056       | 0.644          | 2.3420±4.42           | 0.590        | 2.10             |

### Fine-Tuning Comparison (LLaVA-med)

The table below provides a detailed comparison of LLaVA-med fine-tuning methods:

| Model      | Training Method | Template Acc | Template Posix       | Spelling Acc | Spelling Posix       | Paraphrase Acc | Paraphrase Posix      | Model Avg Acc | Model Avg Posix  |
|------------|-----------------|--------------|----------------------|--------------|----------------------|----------------|-----------------------|---------------|------------------|
| LLAVA-med  | Zero-shot      | 0.755        | 0.0208±0.046         | 0.716        | 0.3069±0.17          | 0.619          | 1.0163±0.216          | 0.696         | 0.448            |
| LLAVA-med  | CE Loss        | 0.746        | 0.0293±0.0519        | 0.684        | 0.3424±0.1748        | 0.584          | 1.0664±0.2076         | 0.6714        | 0.479            |
| LLAVA-med  | **Posix Loss** | **0.763**    | **0.0178±0.052**     | **0.749**    | **0.2482±0.1427**     | **0.677**     | **0.8988±0.2002**      | **0.729**    | **0.388**        |

*Best results are highlighted in bold.*

These results demonstrate that fine-tuning with the POSIX Loss significantly improves both accuracy and consistency, particularly in challenging paraphrase scenarios.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-enabled GPU 
- Other required libraries as listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sarim-MBZUAI/prompt_sensitivity.git
   cd prompt_sensitivity
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```




## Contributing

Contributions, suggestions, and bug reports are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or further discussion, please contact:
- [Sarim Hashmi](mailto:sarim.hashmi@mbzuai.ac.ae)


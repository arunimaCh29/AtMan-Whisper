# AtMan Adaptation for Whisper ASR

This project implements and evaluates the adaptation of the AtMan[1] explainability method to OpenAI's Whisper model for automatic speech recognition (ASR).

## Overview

The AtMan method has been modified to work with Whisper's architecture to provide insights into how the model processes and transcribes audio inputs. The implementation allows for analyzing the importance of different temporal regions in the audio for specific token predictions.

## Key Modifications

### Changes in the Whisper Model 

1. Added support for attention manipulation in Whisper's architecture:
   - Modified MultiHeadAttention to accept suppression parameters
   - Updated ResidualAttentionBlock to propagate suppression information

2. Extended DecodingOptions to include:
   - `perturbation`: Boolean flag to enable attention manipulation
   - `perturbation_tokens`: Dictionary mapping token indices to their suppression factors

### Analysis Features

The implementation provides two main types of analysis:

1. Token-level Analysis:
   - Measures impact of temporal suppression on individual token predictions
   - Supports different suppression window sizes and suppression factors

2. Sentence-level Analysis:
   - Evaluates overall transcription robustness
   - Analyzes cross-entropy loss patterns for the full sentence after suppression for varying window sizes and suppression factors

## Visualization Tools

The project includes visualization tools for:
- Audio waveforms with transcription overlay
- Log-mel spectrograms
- Token-level loss patterns
- Temporal importance heatmaps
- Sentence-level loss patterns

## Interpretation Guide

### Loss Patterns
- Higher loss values indicate regions more critical for token prediction
- Sudden spikes in loss suggest important temporal regions

### Suppression Effects
- Strong impact: Critical region for token prediction
- Minimal impact: Region less relevant for prediction
- Variable impact: Complex dependencies in processing

### Alignment Visualization
- The visualization includes a plot showing cross entropy loss against temporal frames for individual tokens, highlighting how each frame influences token prediction. A shaded overlay indicates the temporal region where the token is expected to occur, helping identify the most significant frames for that token's recognition.

## Usage Examples
This implementation includes a pre-packaged Whisper model, eliminating the need for separate installation.

The analysis capabilities are demonstrated in the 'whisper-tiny.ipynb' Jupyter notebook, which covers:
1. Analysis of individual token behaviors
2. Full sentence-level examination
3. Mapping of temporal significance
4. Comparative evaluation of different suppression configurations

### Example Code for Perturbation Analysis

```python
# Load the model
model = whisper.load_model("tiny")

# Prepare audio input
mel = whisper.log_mel_spectrogram(audio)

# 1. Basic transcription without perturbation
options = DecodingOptions(
    task="transcribe",
    language="english"
)
result = model.decode(mel, options)
baseline_text = result.text
baseline_logits = result.logits

# 2. Transcription with perturbation
# Create perturbation dictionary mapping token indices to suppression factors
suppression_dict = {
    5: 0.5,  # Suppress token at index 5 by 50%
    6: 0.5  # Suppress token at index 6 by 50%
}

options_with_perturbation = DecodingOptions(
    task="transcribe",
    language="english",
    perturbation=True,  # Enable perturbation
    perturbation_tokens=suppression_dict
)
perturbed_result = model.decode(mel, options_with_perturbation)
perturbed_logits = perturbed_result.logits
# Calculate cross entropy loss

# Calculate cross entropy loss between original and perturbed logits

cross_entropy_loss = F.cross_entropy(
    perturbed_logits.view(-1, perturbed_logits.size(-1)), 
    baseline_logits.view(-1, baseline_logits.size(-1)).argmax(dim=-1),
    reduction='mean'
)


```

## Key Functions and Parameters

1. **DecodingOptions**:
   - `perturbation`: Boolean flag to enable attention manipulation
   - `perturbation_tokens`: Dictionary mapping temporal indices to suppression factors (0.0 to 1.0)

2. **Model Methods**:
   - `model.decode()`: Main decoding function that accepts perturbation parameters
   - `model.encoder()`: Encoder that processes perturbations in attention layers

3. **Visualization Functions**:
   ```python
   # Plot token-level loss patterns
   plot_logits_per_word(data, tokens, suppression_len, suppression_factor)
   
   # Plot temporal importance heatmap
   plot_temporal_importance(results, text)
   
   # Plot logits per word with alignment
   plot_logits_per_word_with_alignment(data, decoded_result, text, suppression_len, suppression_factor, audio, model, output_dir)

   # Plot factor comparison
   plot_factor_comparison(results, target_len=5)   

   # Plot parameter comparisons
   plot_parameter_comparisons(results, target_len=5)
   ```

## Implementation Details

### Key Modified Files

1. **whisper/model.py**:
   - `MultiHeadAttention.forward()`: Added support for attention manipulation
   - `MultiHeadAttention.qkv_attention()`: Implemented suppression mechanism
   - `ResidualAttentionBlock.forward()`: Modified to propagate suppression parameters
   - `AudioEncoder.forward()`: Updated to handle perturbations

2. **whisper/decoding.py**:
   - `DecodingOptions`: Added perturbation-related parameters
   - `DecodingTask.__init__()`: Initialized perturbation parameters
   - `DecodingTask._main_loop()`: Modified to handle perturbations during decoding

3. **whisper_notebooks/whisper-tiny.ipynb**:
   - Contains example implementations and analysis workflows
   - Demonstrates visualization techniques
   - Shows different perturbation scenarios

### Key Functions Flow

1. **Perturbation Entry Point**:
   ```python
   model.decode(mel, options_with_perturbation)
   ```

2. **Attention Manipulation Path**:
   ```
   decode() -> DecodingTask.run() -> _main_loop() -> 
   inference.logits() -> model.decoder() -> MultiHeadAttention.qkv_attention()
   ```

3. **Visualization Flow**:
   ```
   analyze_temporal_importance() -> collect_logits() -> 
   plot_logits_per_word() / plot_logits_per_word_with_alignment()
   ```

## Acknowledgements

This work was conducted as a research and development project for the Master's in Autonomous Systems program at Hochschule Bonn-Rhein-Sieg University of Applied Sciences, supervised by Prof. Dr. Sebastian Houben and Roman Bartolosch (Fraunhofer FKIE).

## References

[1] B. Deiseroth, M. Deb, S. Weinbach, M. Brack, P. Schramowski, and K. Kersting, "Atman: Understanding transformer predictions through memory efficient attention manipulation," in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36, Curran Associates, Inc., 2023, pp. 63437-63460. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2023/file/c83bc020a020cdeb966ed10804619664-Paper-Conference.pdf

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in International Conference on Machine Learning. PMLR, 2023, pp. 28492-28518.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc., 2017, pp. 5998-6008.

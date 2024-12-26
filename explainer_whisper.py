import torch.nn as nn
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .utils import chunks, dict_to_json
from .conceptual_suppression import ConceptualSuppression

class WhisperExplainer:
    def __init__(self, model: WhisperForConditionalGeneration, processor: WhisperProcessor, device,
                 suppression_factor=0.1, conceptual_suppression_threshold=0.6,
                 modify_suppression_factor_based_on_cossim=True,
                 multiplicative=True, layers=None,
                 manipulate_attn_scores_after_scaling: bool = False):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.layers = layers
        self.suppression_factor = suppression_factor
        self.conceptual_suppression_threshold = conceptual_suppression_threshold
        self.modify_suppression_factor_based_on_cossim = modify_suppression_factor_based_on_cossim
        self.manipulate_attn_scores_after_scaling = manipulate_attn_scores_after_scaling

        if conceptual_suppression_threshold is not None:
            assert 0 <= conceptual_suppression_threshold <= 1., "Conceptual suppression threshold must be between 0 and 1."

    def preprocess_audio(self, audio_input):
        """Preprocess audio input for Whisper."""
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        return inputs

    def collect_logits_by_manipulating_attention(self, audio_input, target_text, explain_indices=None,
                                                  max_batch_size=1, configs=None, save_configs_as=None,
                                                  save_configs_only=False):
        """Run forward passes by suppressing tokens and collect output logits."""
        inputs = self.preprocess_audio(audio_input).to(self.device)
        target_token_ids = self.processor.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        prompt_length = inputs.input_features.size(1)

        if explain_indices is None:
            explain_indices = list(range(prompt_length))

        total_seq_len = prompt_length + target_token_ids.size(1)
        target_token_indices = list(range(prompt_length, total_seq_len))

        if configs is None:
            configs = self.get_default_configs_for_forward_passes(
                prompt_explain_indices=explain_indices,
                target_token_indices=target_token_indices
            )
        else:
            assert configs[0] == {"suppression_token_index": [-1], "suppression_factor": [1.0]}, \
                "First config must represent unaltered attention manipulation."

        if save_configs_as is not None:
            dict_to_json(configs, save_configs_as)
            print("Saved configs to:", save_configs_as)

        if save_configs_only:
            return configs

        batches = list(chunks(configs, n=max_batch_size))
        all_logits = self.collect_logits(batches, inputs, target_token_ids)

        return self.compile_results_from_configs_and_logits(all_logits, configs, target_token_ids, target_token_indices, prompt_length, explain_indices)

    def collect_logits(self, batches, inputs, target_token_ids):
        """Collect output logits by running forward passes with modified attention."""
        all_logits = []
        for batch in batches:
            suppression_token_indices = [x['suppression_token_index'] for x in batch]
            suppression_factors = [x['suppression_factor'] for x in batch]

            self.model.suppression_factors = suppression_factors
            self.model.suppression_token_indices = suppression_token_indices
            self.model.manipulate_attn_scores_after_scaling = self.manipulate_attn_scores_after_scaling
            self.model.layers = self.layers

            outputs = self.model(**inputs, decoder_input_ids=target_token_ids)
            logits = outputs.logits

            all_logits.append(logits.cpu())

        return torch.cat(all_logits, dim=0)

    def compile_results_from_configs_and_logits(self, logits, configs, target_token_ids, target_token_indices, prompt_length, explain_indices):
        """Compile results into a dictionary with logits and metadata."""
        results = {
            "original_logits": logits[0].cpu(),
            "target_token_ids": target_token_ids.cpu().tolist(),
            "target_token_indices": target_token_indices,
            "prompt_length": prompt_length,
            "prompt_explain_indices": explain_indices,
            "suppressed_chunk_logits": []
        }

        for idx, config in enumerate(configs[1:], start=1):
            results["suppressed_chunk_logits"].append({
                "suppression_token_indices": config.get("suppression_token_index", []),
                "logits": logits[idx].cpu()
            })

        return results

    def get_default_configs_for_forward_passes(self, prompt_explain_indices, target_token_indices):
        """Generate default suppression configs for forward passes."""
        configs = [{"suppression_token_index": [-1], "suppression_factor": [1.0]}]

        for idx in prompt_explain_indices + target_token_indices:
            configs.append({
                "suppression_token_index": [idx],
                "suppression_factor": [self.suppression_factor]
            })

        return configs

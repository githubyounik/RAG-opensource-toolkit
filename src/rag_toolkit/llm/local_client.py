"""Local LLM chat client backed by a HuggingFace causal language model."""

from __future__ import annotations

from rag_toolkit.llm.base import ChatLLMClient, LLMResponse


class LocalChatClient(ChatLLMClient):
    """Run chat completions using a local HuggingFace causal LM.

    The model is loaded once at construction time.  Any instruction-tuned
    model with a chat template works, e.g. ``Qwen/Qwen2.5-0.5B-Instruct``
    or ``meta-llama/Llama-3.2-1B-Instruct``.

    Parameters
    ----------
    model:
        HuggingFace model ID or absolute local path.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    max_length:
        Hard cap on the total sequence length (prompt + generated tokens).
        Only used as a fallback when ``max_tokens`` is not passed per call.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        max_length: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for LocalChatClient. "
                "Install them with: pip install torch transformers"
            ) from exc

        self.model = model
        self.max_length = max_length

        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self._model.eval()
        self._model.to(self._device)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Run one local chat completion and return normalized text."""
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # apply_chat_template formats messages into the model's expected format
        # and returns the full prompt string (or token IDs when tokenize=True)
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        prompt_length = input_ids.shape[1]
        new_tokens = max_tokens if max_tokens is not None else (self.max_length - prompt_length)
        new_tokens = max(1, new_tokens)

        do_sample = temperature > 0.0
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **generate_kwargs)

        # decode only the newly generated tokens, not the prompt
        generated_ids = output_ids[0][prompt_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        completion_tokens = len(generated_ids)
        return LLMResponse(
            text=text,
            usage={
                "prompt_tokens": prompt_length,
                "completion_tokens": completion_tokens,
            },
        )

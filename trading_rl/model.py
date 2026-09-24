"""Unsloth model wrapper for GRPO training and inference (requires a CUDA GPU + unsloth)."""

from unsloth import FastVisionModel  # must be imported before transformers / trl

import torch
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def plot_training_rewards(log_history):
    import plotly.graph_objects as go
    entries = [e for e in log_history if "reward" in e]
    fig = go.Figure(go.Scatter(
        x=[e["step"] for e in entries], y=[e["reward"] for e in entries],
        mode='lines+markers', name='Reward',
        line=dict(color='royalblue', width=2), marker=dict(size=6),
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.update_layout(title='Training Reward vs Steps', xaxis_title='Step',
                      yaxis_title='Mean Reward', hovermode='x unified', template='plotly_dark')
    fig.show()


class Unsloth:
    """FastVisionModel + LoRA. `FastVisionModel` is Unsloth's unified loader (also for text models)."""

    def __init__(self, model_name, lora_rank=32, max_seq_length=4096, load_in_4bit=True,
                 fast_inference=False, max_prompt_length=512,
                 lora_adapter_path=None, adapter_trainable=True, hf_token=None):
        """
        lora_adapter_path: HF repo / local path of an existing adapter to load (continue
                           training, or evaluate with adapter_trainable=False). If None, a
                           fresh LoRA adapter is created.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=fast_inference,
        )

        if lora_adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, lora_adapter_path, token=hf_token, is_trainable=adapter_trainable)
        else:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                r=lora_rank,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=lora_rank * 2,
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

    def generate(self, prompt, temperature=1.0, max_new_tokens=None, stream=False):
        """Generate a completion for a single user prompt; returns only the new text."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(images=None, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens or self.max_seq_length - self.max_prompt_length,
                streamer=TextStreamer(self.tokenizer, skip_prompt=True) if stream else None,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def hf_repo_name(self, steps):
        return f"{self.model_name.split('/')[-1].lower()}-v1-{steps}"

    def train(self, steps, reward_functions, dataset, *,
              output_dir="outputs", save_dir="grpo_saved_lora",
              hub_model_id=None, hf_token=None, hub_strategy="end",
              save_steps=None, report_to="none", run_name=None,
              resume_from_checkpoint=False, plot=True, **grpo_overrides):
        """Run GRPO, save the LoRA adapter to save_dir and optionally push it to the HF Hub.

        hub_model_id: HF repo to push to (requires hf_token). hub_strategy="checkpoint"
                      pushes every checkpoint so an interrupted run can be resumed.
        grpo_overrides: any other GRPOConfig field (e.g. learning_rate, num_generations).
        """
        max_prompt_length = self.max_prompt_length + 1
        push = bool(hub_model_id and hf_token)

        config = dict(
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_steps=max(1, int(0.1 * steps)),
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=2,
            max_prompt_length=max_prompt_length,
            max_completion_length=self.max_seq_length - max_prompt_length,
            max_steps=steps,
            save_steps=save_steps or steps,
            report_to=report_to,
            run_name=run_name,
            output_dir=output_dir,
            push_to_hub=push,
            hub_model_id=hub_model_id if push else None,
            hub_strategy=hub_strategy,
            hub_token=hf_token if push else None,
        )
        config.update(grpo_overrides)

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.max_length = None

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
            args=GRPOConfig(**config),
            train_dataset=dataset,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
        if plot:
            plot_training_rewards(trainer.state.log_history)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"LoRA adapter saved to {save_dir}/")

        if push:
            self.model.push_to_hub(hub_model_id, token=hf_token, save_method="lora")
            self.tokenizer.push_to_hub(hub_model_id, token=hf_token)
            print(f"Pushed to https://huggingface.co/{hub_model_id}")
        return trainer

    def unload(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        print("Model unloaded and GPU cache cleared.")

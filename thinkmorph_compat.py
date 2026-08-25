"""Compatibility layer for the ThinkMorph evaluator used by VBVR-Pro."""

from __future__ import annotations

from copy import deepcopy

import torch
from data.data_utils import pil_img2rgb
from inferencer import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
    InterleaveInferencer,
)
from PIL import Image


class ThinkMorphInterleaveInferencer(InterleaveInferencer):
    """Use the CFG-context updates from the released VBVR-Pro evaluator."""

    def interleave_inference(
        self,
        input_lists,
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        max_rounds=10,
        system_prompt=None,
    ) -> list[str | Image.Image]:
        del enable_taylorseer
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think or system_prompt:
                if system_prompt is None:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                output_list.append(
                    self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                )
                return output_list

            rounds = 0
            while rounds < max_rounds:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)
                gen_context = self.update_context_text(gen_text, gen_context)
                cfg_img_context = self.update_context_text(gen_text, cfg_img_context)
                if "<image_start>" not in gen_text:
                    break

                image = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                output_list.append(image)
                image_input = self.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = self.update_context_image(image_input, gen_context, vae=not understanding_output)
                cfg_text_context = deepcopy(gen_context)
                rounds += 1

        return output_list

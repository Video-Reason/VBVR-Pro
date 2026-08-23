# VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

<div align="center">

<p align="center">
    <a href="https://video-reason.com/?v=pro" target="_blank">
        <img alt="Project Page" src="https://img.shields.io/badge/Project%20-%20Homepage-4285F4" height="20" />
    </a>
    <a href="https://huggingface.co/papers/2602.20159" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VBVR_Pro-red?logo=arxiv" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-Pro" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Training_&_Inference-VBVR_Pro-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-Pro-Bench" target="_blank">
        <img alt="Eval Code" src="https://img.shields.io/badge/Evaluation_code-VBVR_Pro_Bench-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Dataset-Video-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Dataset-Image-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench/tree/main" target="_blank">
        <img alt="Bench Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Bench-Data-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://video-reason.com/pro/bench/#leaderboard" target="_blank">
        <img alt="Leaderboard" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Bench-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
</p>

</div>

## Overview

Native visual reasoning, i.e., reasoning through visual generation, has recently emerged as a promising direction for studying visual intelligence beyond language. Yet progress remains bottlenecked by the lack of scalable training tasks, reliable feedback, and controlled comparisons across generative substrates. In this work, we introduce **VBVR-Pro**, a closed-loop testbed that makes native visual reasoning through generation trainable, verifiable, optimizable, and experimentally controllable. **1) Task scaling.** VBVR-Pro turns visual reasoning into a controlled task space of *300* procedurally generated tasks. Models trained on VBVR-Pro show strong transfer beyond the proposed suite across *six* held-out visual reasoning benchmarks such as RISE-Video, MME-CoF-Pro, and BabyVision. Further analysis validates that these gains reflect visual reasoning rather than instruction-pattern fitting. **2) Verifiable rewards.** VBVR-Pro provides verifiable reward scorers for task-grounded evaluation. Through a systematic study of leading MLLMs as judges, we identify recurring failure modes of the prevalent *VLM-as-a-judge* paradigm. In contrast, the proposed scorers are grounded on verifiable task-specific rules, achieve fine-grained alignment with human judgments. Importantly, they serve as reliable reward signals for large-scale multi-task reinforcement learning and demonstrate stronger post-RL performance across visual reasoning tasks. **3) Mechanism study.** VBVR-Pro enables controlled modality studies across more than *30* image, video, and interleaved generators. Our analysis shows that video generation remains strongest for tasks requiring persistent spatiotemporal state tracking, while interleaved generation provides a compute-efficient alternative by externalizing intermediate visual states. Critically, ablations and probing confirm the presence of vision-native trajectories, that are a more crucial substrate than explicit linguistic chains of thought for visual reasoning. We release all data, models, scorers, and code to facilitate future research.


## Models Zoo

<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th>Model</th>
      <th>Base Architecture</th>
      <th>Other Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="3" align="left">Image Generation Models</th></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-BAGEL"><strong>VBVR-Pro-BAGEL</strong></a></td><td>BAGEL-7B-MoT</td><td>Complete model</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-FLUX2-dev">VBVR-Pro-FLUX2-dev</a></td><td>FLUX.2-dev</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-FLUX2-dev-diffsynth">VBVR-Pro-FLUX2-dev-diffsynth</a></td><td>FLUX.2-dev</td><td>LoRA model, DiffSynth format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Qwen-Image-Edit">VBVR-Pro-Qwen-Image-Edit</a></td><td>Qwen-Image-Edit-2511</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Qwen-Image-Edit-diffsynth">VBVR-Pro-Qwen-Image-Edit-diffsynth</a></td><td>Qwen-Image-Edit-2511</td><td>LoRA model, DiffSynth format</td></tr>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="3" align="left">Interleaved Image Generation Models</th></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-ThinkMorph">VBVR-Pro-ThinkMorph</a></td><td>ThinkMorph-7B</td><td>Complete model</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-SenseNova-U1">VBVR-Pro-SenseNova-U1</a></td><td>SenseNova-U1-8B-MoT</td><td>Complete model</td></tr>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="3" align="left">Video Generation Models</th></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-LTX2.3">VBVR-Pro-LTX2.3</a></td><td>LTX-Video-2.3</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-LTX2.3-diffsynth">VBVR-Pro-LTX2.3-diffsynth</a></td><td>LTX-Video-2.3</td><td>LoRA model, DiffSynth format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.1-I2V-14B">VBVR-Pro-Wan2.1-I2V-14B</a></td><td>Wan2.1-I2V-14B-720P</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.1-I2V-14B-diffsynth">VBVR-Pro-Wan2.1-I2V-14B-diffsynth</a></td><td>Wan2.1-I2V-14B-720P</td><td>LoRA model, DiffSynth format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B">VBVR-Pro-Wan2.2-I2V-A14B</a></td><td>Wan2.2-I2V-A14B</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B-diffsynth">VBVR-Pro-Wan2.2-I2V-A14B-diffsynth</a></td><td>Wan2.2-I2V-A14B</td><td>LoRA model, DiffSynth format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B">VBVR-Pro-Wan2.2-TI2V-5B</a></td><td>Wan2.2-TI2V-5B</td><td>Complete model, Diffusers format</td></tr>
    <tr><td><a href="https://huggingface.co/Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B-diffsynth">VBVR-Pro-Wan2.2-TI2V-5B-diffsynth</a></td><td>Wan2.2-TI2V-5B</td><td>LoRA model, DiffSynth format</td></tr>
  </tbody>
</table>

## VBVR-Pro Benchmark Results

<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse; width: 100%; font-size: 12px;">
  <thead>
    <tr>
      <th rowspan="2">Models</th>
      <th rowspan="2">Overall</th>
      <th colspan="6">In-Domain by Category</th>
      <th colspan="6">Out-of-Domain by Category</th>
    </tr>
    <tr>
      <th>Avg.</th><th>Abst.</th><th>Know.</th><th>Perc.</th><th>Spat.</th><th>Trans.</th>
      <th>Avg.</th><th>Abst.</th><th>Know.</th><th>Perc.</th><th>Spat.</th><th>Trans.</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="14" align="left">Image Generation Models</th></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Proprietary Models</th></tr>
    <tr><td>Qwen-Image-2.0</td><td><u>0.313</u></td><td><u>0.248</u></td><td><u>0.269</u></td><td><u>0.196</u></td><td><u>0.225</u></td><td><u>0.170</u></td><td><u>0.132</u></td><td><u>0.378</u></td><td><u>0.341</u></td><td><u>0.235</u></td><td><u>0.391</u></td><td><u>0.384</u></td><td><u>0.080</u></td></tr>
    <tr><td>Seedream-5.0-Pro</td><td><strong>0.557</strong></td><td><strong>0.485</strong></td><td><strong>0.518</strong></td><td><strong>0.312</strong></td><td><strong>0.509</strong></td><td><strong>0.401</strong></td><td><strong>0.217</strong></td><td><strong>0.629</strong></td><td><strong>0.507</strong></td><td><strong>0.455</strong></td><td><strong>0.661</strong></td><td><strong>0.559</strong></td><td><strong>0.202</strong></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Open-source Models</th></tr>
    <tr><td>BAGEL-7B-MoT</td><td>0.089</td><td><u>0.066</u></td><td>0.039</td><td><u>0.085</u></td><td>0.067</td><td>0.046</td><td>0.027</td><td>0.111</td><td><strong>0.201</strong></td><td>0.031</td><td>0.073</td><td>0.028</td><td><strong>0.121</strong></td></tr>
    <tr><td>FLUX.2-dev</td><td><strong>0.157</strong></td><td><strong>0.108</strong></td><td><u>0.088</u></td><td><strong>0.109</strong></td><td><u>0.072</u></td><td><u>0.100</u></td><td><strong>0.066</strong></td><td><strong>0.206</strong></td><td><u>0.197</u></td><td><strong>0.165</strong></td><td><strong>0.184</strong></td><td><strong>0.241</strong></td><td>0.077</td></tr>
    <tr><td>Qwen-Image-Edit</td><td><u>0.134</u></td><td><strong>0.108</strong></td><td><strong>0.092</strong></td><td>0.082</td><td><strong>0.100</strong></td><td><strong>0.109</strong></td><td><u>0.056</u></td><td><u>0.159</u></td><td>0.176</td><td><u>0.063</u></td><td><u>0.141</u></td><td><u>0.182</u></td><td><u>0.082</u></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Strong Baselines</th></tr>
    <tr><td>VBVR-Pro-BAGEL</td><td>0.172</td><td>0.168</td><td>0.199</td><td>0.105</td><td>0.110</td><td>0.213</td><td>0.055</td><td>0.176</td><td>0.254</td><td>0.104</td><td>0.148</td><td>0.015</td><td><u>0.145</u></td></tr>
    <tr><td>VBVR-Pro-FLUX.2</td><td><strong>0.407</strong></td><td><strong>0.484</strong></td><td><strong>0.483</strong></td><td><strong>0.323</strong></td><td><strong>0.367</strong></td><td><strong>0.449</strong></td><td><strong>0.336</strong></td><td><strong>0.330</strong></td><td><strong>0.361</strong></td><td><strong>0.272</strong></td><td><strong>0.255</strong></td><td><strong>0.454</strong></td><td>0.128</td></tr>
    <tr><td>VBVR-Pro-Qwen-Image</td><td><u>0.322</u></td><td><u>0.332</u></td><td><u>0.298</u></td><td><u>0.217</u></td><td><u>0.193</u></td><td><u>0.431</u></td><td><u>0.222</u></td><td><u>0.311</u></td><td><u>0.341</u></td><td><u>0.239</u></td><td><u>0.233</u></td><td><u>0.413</u></td><td><strong>0.181</strong></td></tr>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="14" align="left">Interleaved Image Generation Models</th></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Proprietary Models</th></tr>
    <tr><td>GPT-Image-2</td><td><u>0.507</u></td><td><u>0.428</u></td><td><u>0.456</u></td><td><u>0.318</u></td><td><u>0.428</u></td><td><u>0.206</u></td><td><strong>0.300</strong></td><td><u>0.587</u></td><td><u>0.398</u></td><td><u>0.413</u></td><td><u>0.633</u></td><td><u>0.480</u></td><td><strong>0.303</strong></td></tr>
    <tr><td>Nano Banana Pro</td><td><strong>0.564</strong></td><td><strong>0.480</strong></td><td><strong>0.518</strong></td><td><strong>0.422</strong></td><td><strong>0.512</strong></td><td><strong>0.285</strong></td><td><u>0.174</u></td><td><strong>0.648</strong></td><td><strong>0.553</strong></td><td><strong>0.499</strong></td><td><strong>0.657</strong></td><td><strong>0.585</strong></td><td><u>0.220</u></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Open-source Models</th></tr>
    <tr><td>ThinkMorph-7B</td><td>0.154</td><td>0.113</td><td>0.100</td><td>0.082</td><td>0.101</td><td>0.148</td><td>0.031</td><td>0.195</td><td>0.176</td><td>0.166</td><td>0.163</td><td>0.253</td><td>0.103</td></tr>
    <tr><td>VBVR-SenseNova-U1</td><td><u>0.408</u></td><td><u>0.469</u></td><td><u>0.356</u></td><td><u>0.313</u></td><td><u>0.373</u></td><td><strong>0.386</strong></td><td><strong>0.477</strong></td><td><u>0.347</u></td><td><u>0.291</u></td><td><u>0.317</u></td><td><u>0.275</u></td><td><u>0.480</u></td><td><u>0.238</u></td></tr>
    <tr><td>SenseNova-U1-8B-MoT</td><td><strong>0.565</strong></td><td><strong>0.533</strong></td><td><strong>0.501</strong></td><td><strong>0.395</strong></td><td><strong>0.544</strong></td><td><u>0.355</u></td><td><u>0.349</u></td><td><strong>0.597</strong></td><td><strong>0.448</strong></td><td><strong>0.495</strong></td><td><strong>0.533</strong></td><td><strong>0.717</strong></td><td><strong>0.401</strong></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Strong Baselines</th></tr>
    <tr><td>VBVR-Pro-ThinkMorph</td><td><u>0.373</u></td><td><u>0.402</u></td><td><u>0.403</u></td><td><u>0.344</u></td><td><u>0.238</u></td><td><u>0.454</u></td><td><u>0.184</u></td><td><u>0.344</u></td><td><u>0.367</u></td><td><u>0.224</u></td><td><u>0.238</u></td><td><u>0.535</u></td><td><u>0.257</u></td></tr>
    <tr><td>VBVR-Pro-SenseNova-U1</td><td><strong>0.638</strong></td><td><strong>0.811</strong></td><td><strong>0.648</strong></td><td><strong>0.695</strong></td><td><strong>0.621</strong></td><td><strong>0.770</strong></td><td><strong>0.541</strong></td><td><strong>0.464</strong></td><td><strong>0.480</strong></td><td><strong>0.328</strong></td><td><strong>0.344</strong></td><td><strong>0.558</strong></td><td><strong>0.408</strong></td></tr>
    <tr style="border-top: 4px solid #6b7280; background-color: #e5e7eb;"><th colspan="14" align="left">Video Generation Models</th></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Proprietary Models</th></tr>
    <tr><td>Veo 3.1</td><td>0.309</td><td>0.312</td><td><u>0.275</u></td><td>0.299</td><td>0.252</td><td>0.267</td><td>0.157</td><td>0.305</td><td><u>0.305</u></td><td>0.233</td><td>0.252</td><td><u>0.312</u></td><td>0.219</td></tr>
    <tr><td>Kling V3</td><td><u>0.392</u></td><td><u>0.356</u></td><td>0.213</td><td><u>0.326</u></td><td><u>0.320</u></td><td><u>0.355</u></td><td><u>0.229</u></td><td><u>0.427</u></td><td>0.294</td><td><strong>0.564</strong></td><td><u>0.375</u></td><td>0.242</td><td><u>0.412</u></td></tr>
    <tr><td>SeedDance 2.0</td><td><strong>0.499</strong></td><td><strong>0.451</strong></td><td><strong>0.338</strong></td><td><strong>0.361</strong></td><td><strong>0.353</strong></td><td><strong>0.468</strong></td><td><strong>0.308</strong></td><td><strong>0.547</strong></td><td><strong>0.369</strong></td><td><u>0.511</u></td><td><strong>0.478</strong></td><td><strong>0.538</strong></td><td><strong>0.532</strong></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Open-source Models</th></tr>
    <tr><td>HunyuanVideo-I2V</td><td>0.054</td><td>0.054</td><td>0.023</td><td>0.064</td><td>0.015</td><td>0.084</td><td>0.032</td><td>0.053</td><td>0.088</td><td>0.014</td><td>0.028</td><td>0.062</td><td>0.055</td></tr>
    <tr><td>CogVideoX1.5-5B-I2V</td><td>0.085</td><td>0.100</td><td>0.061</td><td>0.118</td><td>0.069</td><td>0.092</td><td>0.060</td><td>0.070</td><td>0.125</td><td>0.038</td><td>0.051</td><td>0.040</td><td>0.024</td></tr>
    <tr><td>Wan2.1-I2V-14B</td><td>0.100</td><td>0.105</td><td>0.052</td><td>0.125</td><td>0.091</td><td>0.102</td><td>0.052</td><td>0.095</td><td>0.112</td><td>0.073</td><td>0.071</td><td>0.123</td><td>0.044</td></tr>
    <tr><td>Wan2.2-TI2V-5B</td><td>0.094</td><td>0.066</td><td>0.029</td><td>0.073</td><td>0.050</td><td>0.083</td><td>0.031</td><td>0.122</td><td>0.156</td><td>0.052</td><td>0.106</td><td>0.063</td><td>0.099</td></tr>
    <tr><td>Wan2.2-I2V-14B-720P</td><td><u>0.182</u></td><td><u>0.157</u></td><td><u>0.082</u></td><td><u>0.131</u></td><td><u>0.110</u></td><td><u>0.161</u></td><td><u>0.156</u></td><td><u>0.207</u></td><td><u>0.224</u></td><td><u>0.139</u></td><td><u>0.140</u></td><td><u>0.195</u></td><td><u>0.273</u></td></tr>
    <tr><td>LTX2.3-I2AV</td><td>0.112</td><td>0.106</td><td>0.062</td><td>0.109</td><td>0.070</td><td>0.133</td><td>0.055</td><td>0.119</td><td>0.161</td><td>0.135</td><td>0.086</td><td>0.091</td><td>0.050</td></tr>
    <tr><td>VBVR-Wan2.2</td><td><strong>0.517</strong></td><td><strong>0.548</strong></td><td><strong>0.237</strong></td><td><strong>0.499</strong></td><td><strong>0.334</strong></td><td><strong>0.566</strong></td><td><strong>0.591</strong></td><td><strong>0.486</strong></td><td><strong>0.310</strong></td><td><strong>0.343</strong></td><td><strong>0.345</strong></td><td><strong>0.732</strong></td><td><strong>0.684</strong></td></tr>
    <tr style="background-color: #dbeafe;"><th colspan="14" align="left">Strong Baselines</th></tr>
    <tr><td>VBVR-Pro-LTX2.3</td><td>0.425</td><td>0.527</td><td>0.409</td><td>0.510</td><td>0.346</td><td>0.460</td><td>0.390</td><td>0.324</td><td>0.381</td><td>0.108</td><td>0.201</td><td>0.477</td><td>0.386</td></tr>
    <tr><td>VBVR-Pro-Wan2.1-I2V-14B</td><td><u>0.562</u></td><td><u>0.730</u></td><td><u>0.617</u></td><td><u>0.580</u></td><td><u>0.452</u></td><td><u>0.676</u></td><td><u>0.623</u></td><td><u>0.395</u></td><td><u>0.410</u></td><td><u>0.305</u></td><td><u>0.230</u></td><td><u>0.617</u></td><td><u>0.439</u></td></tr>
    <tr><td>VBVR-Pro-Wan2.2-TI2V-5B</td><td>0.470</td><td>0.641</td><td>0.528</td><td>0.556</td><td>0.373</td><td>0.565</td><td>0.557</td><td>0.300</td><td>0.333</td><td>0.127</td><td>0.161</td><td>0.505</td><td>0.409</td></tr>
    <tr><td>VBVR-Pro-Wan2.2-I2V-14B</td><td><strong>0.670</strong></td><td><strong>0.808</strong></td><td><strong>0.632</strong></td><td><strong>0.685</strong></td><td><strong>0.556</strong></td><td><strong>0.751</strong></td><td><strong>0.636</strong></td><td><strong>0.532</strong></td><td><strong>0.479</strong></td><td><strong>0.418</strong></td><td><strong>0.350</strong></td><td><strong>0.679</strong></td><td><strong>0.690</strong></td></tr>
  </tbody>
</table>

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) to manage the unified Python 3.10 environment.

> uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/#installing-uv>

```bash
git clone https://github.com/Video-Reason/VBVR-Pro.git
cd VBVR-Pro/
uv sync --extra cu124 # or one of [cu118|cu121|cu124|cu126|cu128|cu129]
source .venv/bin/activate
```

Select the extra for the CUDA wheel build you want; the extras are mutually exclusive.
For faster attention on BAGEL and ThinkMorph, install the optional FlashAttention extension:

```bash
uv sync --extra cu124 --extra flash-attn
```

## Download Models

### Merged models (self-contained, recommended)

The merged (complete) models do not require separate base model downloads:

```bash
# Image generation
hf download Video-Reason/VBVR-Pro-FLUX2-dev --local-dir models/VBVR-Pro-FLUX2-dev
hf download Video-Reason/VBVR-Pro-Qwen-Image-Edit --local-dir models/VBVR-Pro-Qwen-Image-Edit
hf download Video-Reason/VBVR-Pro-BAGEL --local-dir models/VBVR-Pro-BAGEL

# Interleaved image generation
hf download Video-Reason/VBVR-Pro-ThinkMorph --local-dir models/VBVR-Pro-ThinkMorph
hf download Video-Reason/VBVR-Pro-SenseNova-U1 --local-dir models/VBVR-Pro-SenseNova-U1

# Video generation
hf download Video-Reason/VBVR-Pro-LTX2.3 --local-dir models/VBVR-Pro-LTX2.3
hf download Video-Reason/VBVR-Pro-Wan2.1-I2V-14B --local-dir models/VBVR-Pro-Wan2.1-I2V-14B
hf download Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B --local-dir models/VBVR-Pro-Wan2.2-I2V-A14B
hf download Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B --local-dir models/VBVR-Pro-Wan2.2-TI2V-5B
```

### DiffSynth LoRA models (require base model downloads)

For `-diffsynth` releases, download the corresponding base models first:

```bash
export MODELS_DIR="${PWD}/models"
mkdir -p "${MODELS_DIR}"

hf download black-forest-labs/FLUX.2-dev \
  --local-dir "${MODELS_DIR}/black-forest-labs/FLUX.2-dev"
hf download Qwen/Qwen-Image-Edit-2511 \
  --local-dir "${MODELS_DIR}/Qwen/Qwen-Image-Edit-2511"
hf download DiffSynth-Studio/LTX-2.3-Repackage \
  --local-dir "${MODELS_DIR}/DiffSynth-Studio/LTX-2.3-Repackage"
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "${MODELS_DIR}/google/gemma-3-12b-it-qat-q4_0-unquantized"
hf download Wan-AI/Wan2.1-I2V-14B-720P \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.1-I2V-14B-720P"
hf download Wan-AI/Wan2.2-I2V-A14B \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.2-I2V-A14B"
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.2-TI2V-5B"
```

## Inference

All models are invoked through the unified `example.py` CLI:

```bash
# Image editing (FLUX.2 merged)
python example.py \
  --model_path models/VBVR-Pro-FLUX2-dev \
  --image_paths first_frame.png \
  --prompt "Move the red block to the left of the blue block." \
  --output flux_edit.png

# Video generation (Wan2.2 merged)
python example.py \
  --model_path models/VBVR-Pro-Wan2.2-I2V-A14B \
  --image_paths first_frame.png \
  --prompt "The subject walks toward the doorway." \
  --num_frames 81 --width 832 --height 480 \
  --output wan.mp4

# Audio-video generation (LTX-2.3 merged)
python example.py \
  --model_path models/VBVR-Pro-LTX2.3 \
  --image_paths first_frame.png \
  --prompt "The machine starts and makes a quiet mechanical hum." \
  --num_frames 49 --fps 24 \
  --output ltx.mp4

# Interleaved generation with reasoning (ThinkMorph)
python example.py \
  --model_path models/VBVR-Pro-ThinkMorph \
  --image_paths first_frame.png \
  --prompt "Solve the task step by step." \
  --think --max_rounds 10 --num_images 2 \
  --output thinkmorph_outputs

# DiffSynth LoRA (downloads base on first use)
python example.py \
  --model_path models/VBVR-Pro-Qwen-Image-Edit-diffsynth \
  --image_paths first_frame.png \
  --prompt "Place the cup on the empty shelf." \
  --output qwen_lora.png
```

Use `python example.py --list-models` for the full list of supported model types.

## Training

For reproducible SFT data preparation and multi-node training of all model families,
see [`TRAINING.md`](TRAINING.md).

## Download Evaluation Data

```bash
hf download Video-Reason/VBVR-Pro-Bench --repo-type dataset --local-dir data/VBVR-Pro-Bench
```

After generating outputs, evaluate using [VBVR-Pro-Bench](https://github.com/Video-Reason/VBVR-Pro-Bench) and submit results to the [leaderboard](https://video-reason.com/pro/bench/#leaderboard).

## Citation

```bibtex
@article{vbvr_pro2026,
  title={A Scalable and Verifiable Suite for Native Visual Reasoning},
  author={...},
  journal={arXiv preprint arXiv:2602.20159},
  year={2026}
}
```

## Acknowledgements

This project includes code modified from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [BAGEL](https://github.com/bytedance-seed/BAGEL), and [ThinkMorph](https://github.com/ThinkMorph/ThinkMorph). We gratefully acknowledge the authors and contributors for their open research.

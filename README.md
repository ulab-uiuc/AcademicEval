# AcademicEval: Live Long-Context LLM Benchmark


<p align="center">
    <a href="https://arxiv.org/abs/2510.17725">
        <img alt="Build" src="https://img.shields.io/badge/arXiv-2510.17725-red?logo=arxiv">
    </a>
    <a href="https://huggingface.co/datasets/ulab-ai/AcademicEval">
        <img alt="HuggingFace" src="https://img.shields.io/badge/%F0%9F%A4%97-AcademicEval-yellow">
    </a>
    <!-- <a href="xxx">
        <img alt="Build" src="https://img.shields.io/badge/Twitter-black?logo=X">
    </a> -->
    <a href="https://github.com/ulab-uiuc/AcademicEval/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/AcademicEval">
        <img alt="Build" src="https://img.shields.io/github/stars/ulab-uiuc/AcademicEval">
    </a>
    <a href="https://github.com/ulab-uiuc/AcademicEval">
        <img alt="Build" src="https://img.shields.io/github/forks/ulab-uiuc/AcademicEval">
    </a>
    <a href="https://github.com/ulab-uiuc/AcademicEval">
        <img alt="Build" src="https://img.shields.io/github/issues/ulab-uiuc/AcademicEval">
    </a>
</p>




<div align=center> <img src="./figures/model.png" width = 85% height="85%"/> </div>



## News


**[2025.10]** 🌟 AcademicEval was released.


**[2025.09]** 🎉 **AcademicEval was accepted by TMLR 2025.**



## Introduction


We proposed <b><i>AcademicEval</i></b>, a live benchmark for evaluating LLMs over long-context generation tasks. <b><i>AcademicEval</i></b> adopts papers on arXiv to introduce several acadeic writing tasks with long-context inputs, <i>i.e.</i>, <b><i>Title, Abstract, Introduction, Related Work</i></b>, wich covers a wide range of abstraction levels and require no manual labeling. 

Comparing to existing long-context LLM benchmarks, our Comparing to existing long-context LLM benchmarks, our AcademicEval offers flexible length, automatic annotation, hierarchical abstraction, few-shot demonstrations, and live updates without data leakage risks. 



<table class="comparison-table">
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Avg Len</th>
      <th>Automatic Annotation</th>
      <th>Hierarchical Abstraction</th>
      <th>Few-shot Demonstrations</th>
      <th>Live Update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ZeroSCROLLS (Shaham et al., 2023)</td>
      <td>~10K</td>
      <td><span style="color: green;">&#x2713;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td>L-Eval (An et al., 2023)</td>
      <td>~8K</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td>BAMBOO (Dong et al., 2023)</td>
      <td>~16K</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td>LongBench (Bai et al., 2023)</td>
      <td>~8K</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: green;">&#x2713;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td>LooGLE (Li et al., 2023)</td>
      <td>~20K</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td>∞Bench (Zhang et al., 2024)</td>
      <td>~200K</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
      <td><span style="color: red;">&#x2718;</td>
    </tr>
    <tr>
      <td><strong>AcademicEval (ours)</strong></td>
      <td><strong>Flexible</strong></td>
      <td><span style="color: green;">&#x2713;</td>
      <td><span style="color: green;">&#x2713;</td>
      <td><span style="color: green;">&#x2713;</td>
      <td><span style="color: green;">&#x2713;</td>
    </tr>
  </tbody>
</table>


<!-- 🔥❗✅❎ -->

**❗❗❗You can download our collected data at [AcademicEval](https://huggingface.co/datasets/ulab-ai/AcademicEval)**


## 📌Environment Setup

### Python Package

```bash
# python==3.10
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install arxiv
pip install tqdm
pip install rouge_score
pip install textstat
pip install transformers
pip install langchain
pip install PyMuPDF
pip install faiss-gpu
pip install openai==0.28.0
```


### LLM Tokenizers

We additionally need the tokenizer configuration files for LLMs to ensure correct and accurate truncation.
- [Gemma](https://huggingface.co/google/gemma-7b-it)
- [LLaMA](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Qwen](https://huggingface.co/Qwen/Qwen1.5-72B-Chat)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Nous Hermes](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)

You only need to download the tokenizer configuration files for each LLM, no model weight files are needed, because we access LLMs through the API. Please place the downloaded files in "gemma", "llama", "qwen", "mixtral", and "hermes" directories, respectively. 

**❗We have integrated these files in our repository.**



## ⭐Experiments

**❗Note: Since we use the LLM API provided by [together.ai](https://www.together.ai/) to access LLMs, you need to prepare your own API KEY in the "get_llm_response_via_api" function in utils.py**

**❗Please ensure that the AcademicEval is downloaded in the "AcademicEval" directory. The path should be like the following:**

```bash
├── README.md
├── abs_extractor.py
├── bart_score.py
├── construct_relation_graph.py
├── exp_comparison.py
├── main.py
├── model.png
├── refine_graph.py
├── related_extractor.py
├── retrieval.py
├── section_region_extractor.py
├── utils.py
├── gemma
│   ├── ...
├── llama
│   ├── ...
├── qwen
│   ├── ...
├── mixtral
│   ├── ...
├── hermes
│   ├── ...
├── AcademicEval
│   ├── abs_9K
│   ├── abs_28K
│   ├── abs_29K_G
│   ├── intro_8K
│   ├── intro_28K
│   ├── intro_28K_G
│   ├── related_34K
│   ├── related_53K
│   ├── related_53K_G
│   ├── title_10K
│   ├── title_30K
│   └── title_31K_G
```


**Here are some command examples, you can run all the experiments by replacing "llm_model" and "setting", or adding "--rag" and "--retriever"**

### **✅*Title Writing***


#### **title-10K**


```bash
# Standard LLMs
python exp_comparison.py --setting title_10K --llm_model google/gemma-7b-it --cuda 3
python exp_comparison.py --setting title_10K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3
# Long-context LLMs
python exp_comparison.py --setting title_10K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting title_10K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting title_10K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting title_10K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting title_10K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```

#### **title-30K**


```bash
# Long-context LLMs
python exp_comparison.py --setting title_30K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting title_30K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting title_30K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting title_30K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting title_30K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **title-31K-G**


```bash
# Long-context LLMs
python exp_comparison.py --setting title_31K_G --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting title_31K_G --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting title_31K_G --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting title_31K_G --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting title_31K_G --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```

### **✅*Abstract Writing***


#### **abs-9K**


```bash
# Standard LLMs
python exp_comparison.py --setting abs_9K --llm_model google/gemma-7b-it --cuda 3
python exp_comparison.py --setting abs_9K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3
# Long-context LLMs
python exp_comparison.py --setting abs_9K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting abs_9K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting abs_9K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting abs_9K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting abs_9K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **abs-28K**


```bash
# Long-context LLMs
python exp_comparison.py --setting abs_28K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting abs_28K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting abs_28K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting abs_28K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting abs_28K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **abs-29K-G**


```bash
# Long-context LLMs
python exp_comparison.py --setting abs_29K_G --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting abs_29K_G --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting abs_29K_G --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting abs_29K_G --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting abs_29K_G --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```



### **✅*Introduction Writing***


#### **intro-8K**


```bash
# Standard LLMs
python exp_comparison.py --setting intro_8K --llm_model google/gemma-7b-it --cuda 3
python exp_comparison.py --setting intro_8K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3
# Long-context LLMs
python exp_comparison.py --setting intro_8K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting intro_8K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting intro_8K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting intro_8K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting intro_8K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **intro-28K**


```bash
# Long-context LLMs
python exp_comparison.py --setting intro_28K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting intro_28K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting intro_28K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting intro_28K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting intro_28K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```

#### **intro-28K-G**


```bash
# Long-context LLMs
python exp_comparison.py --setting intro_28K_G --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting intro_28K_G --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting intro_28K_G --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting intro_28K_G --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting intro_28K_G --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


### **✅*Related Work Writing***


#### **related-34K**


```bash
# Standard LLMs
python exp_comparison.py --setting related_34K --llm_model google/gemma-7b-it --cuda 3
python exp_comparison.py --setting related_34K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3
# Long-context LLMs
python exp_comparison.py --setting related_34K --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
python exp_comparison.py --setting related_34K --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --cuda 3
python exp_comparison.py --setting related_34K --llm_model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --cuda 3
# RALM
python exp_comparison.py --setting related_34K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting related_34K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **related-53K**


```bash
# RALM
python exp_comparison.py --setting related_53K --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting related_53K --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```


#### **related-53K-G**


```bash
# RALM
python exp_comparison.py --setting related_53K_G --llm_model google/gemma-7b-it --cuda 3 --rag --retriever contriever
python exp_comparison.py --setting related_53K_G --llm_model meta-llama/Llama-3-70b-chat-hf --cuda 3 --rag --retriever contriever
```




## 📍Benchmark Construction


We give a general example for constructing AcademicEval benchmark in this section.

**Note: The initial collection process will be time-consuming**

### Co-author Graph Construction

We first collect a co-author graph via the arXiv API. You should prepare your "YOUR START AUTHOR" in construct_relation_graph.py

Then, run the following command to start BFS. 

```bash
python construct_relation_graph.py
```

### Graph Refine

The collected graph may have many defects. Therefore, we provide a complete pipeline for refining the collected graph (including connectivity detection, chronological split, etc.)

```bash
python refine_graph.py
```


### Live Update


You can refer to `live_update.py` for updating the collected co-author graph.



## Other Awesome Works

- [GoR](https://arxiv.org/abs/2410.11001): Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs. [![[code]](https://img.shields.io/github/stars/ulab-uiuc/GoR)](https://github.com/ulab-uiuc/GoR)

- [Thought Retriever](https://openreview.net/pdf?id=sOSAu0XQcI): Thought-Retriever: Don’t Just Retrieve Raw Data, Retrieve Thoughts. [![[code]](https://img.shields.io/github/stars/ulab-uiuc/Thought-Retriever)](https://github.com/ulab-uiuc/Thought-Retriever)




## Citation

```bibtex
@article{AcademicEval,
  title={AcademicEval: Live Long-Context LLM Benchmark},
  author={Haozhen Zhang and Tao Feng and Pengrui Han and Jiaxuan You},
  journal={arXiv preprint arXiv:2510.17725},
  year={2025}
}
```


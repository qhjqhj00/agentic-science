# Research Paper Topic Analysis Report - 2025-10-13
## Table of Contents
- [Topic 1: Memory Efficiency and Compression](#topic-1-memory-efficiency-and-compression) (3 papers)
- [Topic 2: Legal and Regulatory Applications](#topic-2-legal-and-regulatory-applications) (3 papers)
- [Topic 3: Multimodal Reasoning and Perception](#topic-3-multimodal-reasoning-and-perception) (3 papers)
- [Topic 4: Safety and Bias Mitigation](#topic-4-safety-and-bias-mitigation) (4 papers)
- [Topic 5: Language Model Evaluation and Benchmarking](#topic-5-language-model-evaluation-and-benchmarking) (5 papers)
- [Topic 6: Adaptive and Continuous Learning](#topic-6-adaptive-and-continuous-learning) (4 papers)
- [Topic 7: Specialized Domain Applications](#topic-7-specialized-domain-applications) (9 papers)
- [Topic 8: Sentiment and Emotion Analysis](#topic-8-sentiment-and-emotion-analysis) (3 papers)
- [Topic 9: Automated Code and Test Case Generation](#topic-9-automated-code-and-test-case-generation) (3 papers)
- [Topic 10: Miscellaneous Topics](#topic-10-miscellaneous-topics) (55 papers)

---

## Topic 1: Memory Efficiency and Compression

**Topic Overview**

The research topic of memory efficiency and compression for large language models (LLMs) addresses critical challenges in deploying these models in real-world applications. With the increasing size and complexity of LLMs, managing their memory usage becomes a significant hurdle, especially during tasks that require extended context or reasoning capabilities. Efficient memory management not only reduces computational costs but also enhances model scalability and performance, making it possible to deploy advanced AI systems on devices with limited resources. This topic is crucial for optimizing LLMs in various application scenarios, including healthcare, long-form text processing, and interactive dialogues, where maintaining context and reasoning accuracy is paramount.

**Individual Paper Contributions**

**[Wenjie Du] from [Westlake University and Zhejiang University] and colleagues proposed RLKV, a reinforcement learning-based framework for identifying and preserving 'reasoning heads' within the KV cache. The main contributions of this work are the introduction of the concept of 'reasoning heads' that require full KV cache access to maintain reasoning performance, achieving state-of-the-art compression performance with minimal performance degradation, and demonstrating the criticality of reasoning heads through controlled masking experiments. Experiments on GSM8K, Math500, AIME24, and MBPP datasets showed that RLKV outperformed baselines like H2O, R-KV, and DuoAttention in terms of performance stability across different sparsity levels, memory efficiency, and error mode analysis[^0].**

**[Xin Liu] from [Northeastern University, Shenyang, China] and colleagues introduced Semantic-Anchor Compression (SAC), a method that avoids the need for autoencoding pretraining by directly selecting representative tokens as 'anchor tokens' and augmenting them with dedicated 'anchor embeddings'. SAC modifies the LLM’s attention mechanism to bidirectional, allowing anchor tokens to access information from the entire context. This approach solves the problem of prohibitive computational costs and performance degradation associated with direct processing of long contexts. Experiments on SlimPajama-6B and MRQA datasets demonstrated that SAC achieved better ROUGE-1 F1 and Exact Match (EM) scores compared to Full-FT, LLMLingua-2, ICAE, 500xCompressor, and EPL[^49].**

**[Aneesh Jonelagadda] from [Kaliber AI, Research Division] and colleagues designed and implemented Mnemosyne, an unsupervised, human-inspired long-term memory architecture for edge-based LLMs. Mnemosyne's unique contributions include a graph-structured storage system with substance and redundancy filters, memory committing and pruning mechanisms, and a probabilistic recall algorithm with temporal decay and refresh processes. Additionally, it introduces a core summary module for capturing user-specific long-term details. These features address the limitations of current LLM memory systems, such as fixed context limits and the inability to handle repetitive yet distinct conversations. Experiments on the LoCoMo benchmark dataset showed that Mnemosyne significantly improved J-score and human evaluation win rate over baselines including RAG, Mem0, Zep, MemGPT/LangMem, OpenAI, and Memory-R1[^72].**

**Technical Trends**

The papers under this topic highlight evolving trends towards more sophisticated memory management and compression techniques in LLMs. **Wenjie Du** and colleagues focus on the KV cache, using reinforcement learning to dynamically allocate full and compressed access based on the criticality of reasoning heads. This method represents a shift towards intelligent allocation strategies rather than uniform compression. In contrast, **Xin Liu** and team emphasize the avoidance of extensive pretraining phases by leveraging semantically meaningful anchor tokens and bidirectional attention, streamlining the process while maintaining high performance. Lastly, **Aneesh Jonelagadda** and collaborators propose a graph-based architecture inspired by human memory, incorporating dynamic storage and recall mechanisms to handle long-term memory in edge-based environments, indicating a trend towards integrating biological memory principles into machine learning architectures.

**Datasets and Evaluation**

The main datasets utilized across these studies include:
- GSM8K, Math500, AIME24, and MBPP for evaluating KV cache compression and reasoning tasks.
- SlimPajama-6B and MRQA for assessing context compression capabilities.
- LoCoMo benchmark for testing long-term memory architectures in conversational agents.

Evaluation metrics vary according to the specific focus of each paper:
- Performance comparison across different sparsity levels, memory efficiency, and error mode analysis were used by Wenjie Du and colleagues to assess RLKV.
- ROUGE-1 F1 and Exact Match (EM) were employed by Xin Liu and team to evaluate SAC.
- J-score and human evaluation win rate were utilized by Aneesh Jonelagadda and colleagues to measure the effectiveness of Mnemosyne.

These metrics collectively provide a comprehensive assessment of how well each proposed method manages to compress and efficiently utilize memory without sacrificing performance, which is a central concern in the development of practical and scalable LLMs.

---

## Topic 2: Legal and Regulatory Applications

### Topic Overview

Legal and regulatory applications leverage advanced machine learning techniques, particularly large language models (LLMs), to automate the analysis of legal documents, detect AI-generated content, and improve the performance of models in specialized domains. These applications are critical for enhancing efficiency in legal review processes, ensuring compliance with regulations, and safeguarding against the misuse of AI-generated texts. The research in this area aims to develop methodologies that can efficiently and accurately handle complex legal tasks, which often require deep understanding and nuanced interpretation.

### Individual Paper Contributions

**Hyunji Lee from Technical University of Munich and colleagues** proposed a framework combining Monte Carlo Tree Search (MCTS) with a proxy prompt evaluator for optimizing prompts used in the classification of Terms of Service (ToS) clauses as fair or unfair. The main contributions include the integration of MCTS for efficient exploration of the prompt space, the development of a proxy prompt evaluator to reduce the computational cost of evaluating candidate prompts, and the use of textual gradients for iterative prompt refinement. Experiments on the CLAUDETTE dataset demonstrated improved accuracy and Macro F1 scores compared to baselines like SVM with TF-IDF Vectorizer, Fine-tuned LEGAL-BERT, Zero-Shot, GrIPS, and OPRO[^1].

**Elena Khasanova from Dialpad Inc. and colleagues** introduced DACIP-RC, a Domain Adaptive Continual Instruction Pre-Training approach via reading comprehension on business conversations. This method focuses on enhancing the zero-shot instruction-following capabilities of smaller LLMs across diverse business tasks, such as meeting summarization and action item generation. Key innovations include the application of instruction pre-training specifically on business conversational data and the use of reading comprehension techniques to generate diverse instructional tasks. The experimental setup involved proprietary business conversation transcripts along with QMSUM, PubMedQA, and MediQA-QS datasets, showing significant improvements in F1-Score, ROUGE-2, BERTScore, and Likert-scale evaluations over traditional baselines like LLaMA-3.1-8B-Instruct and LLaMA-3.1-8.B-Internal-NTP[^19].

**Cong Zeng from MBZUAI and colleagues** reframed the task of detecting AI-generated texts as an out-of-distribution (OOD) detection problem, proposing a novel one-class learning approach. The method involves training a text encoder to map texts into a high-dimensional space and learning an OOD decision boundary or score, treating human-written texts as outliers relative to machine-generated texts. Innovations include the use of DeepSVDD and HRN for one-class learning and energy-based methods for score-based learning. Experiments on the DeepFake, M4-multilingual, and RAID datasets showed enhanced AUROC, AUPR, and FPR95 metrics compared to previous binary classification approaches like DetectLLM, DNA-GPT, FastDetectGPT, Glimpse, GPTZero, RADAR, GhostBuster, BiScope, and DeTeCtive[^70].

### Technical Trends

The papers in this topic exhibit a trend towards leveraging specialized techniques to address the challenges of applying LLMs in legal and regulatory contexts. Hyunji Lee's team employed MCTS and a proxy evaluator to optimize prompts for better classification outcomes, focusing on fairness detection in ToS documents. Elena Khasanova's group utilized reading comprehension to adapt smaller LLMs for continual learning in business conversational tasks, aiming to overcome the limitations of both large LLMs and traditional fine-tuning methods. Cong Zeng and collaborators developed a one-class learning framework to detect AI-generated texts, addressing the issue of poor generalization in binary classification models.

### Datasets and Evaluation

- **Datasets**: 
  - CLAUDETTE (Hyunji Lee et al.)
  - Proprietary business conversation transcripts, QMSUM, PubMedQA, MediQA-QS (Elena Khasanova et al.)
  - DeepFake, M4-multilingual, RAID (Cong Zeng et al.)

- **Evaluation Metrics**:
  - Accuracy, Macro F1 (Hyunji Lee et al.)
  - F1-Score, ROUGE-2, BERTScore, Likert-scale evaluation, Pairwise preference (Elena Khasanova et al.)
  - AUROC, AUPR, FPR95 (Cong Zeng et al.)

---

## Topic 3: Multimodal Reasoning and Perception

### Topic Overview

Multimodal reasoning and perception is a critical field in artificial intelligence that integrates different types of data—such as text, images, and video—to enable machines to understand and interact with the world in a more human-like manner. This research topic is vital because it addresses the limitations of unimodal models and aims to create more versatile systems capable of handling complex reasoning tasks that require understanding both textual and visual inputs. Improvements in this area can lead to significant advancements in various applications, including autonomous systems, robotics, and human-computer interaction, by making AI more efficient, accurate, and adaptable to real-world scenarios.

### Individual Paper Contributions

**[Shuang Chen] from [various institutions] and colleagues proposed ARES: Adaptive Reasoning via difficulty-aware token-level Entropy reward Shaping, with main contributions being the use of high-window-entropy (HWE) tokens as triggers for exploration, hierarchical reward design that adapts exploration depth based on task difficulty, and token-adaptive KL design to dynamically allocate thinking budgets. These innovations address the issues of verbose long-thought outputs increasing inference costs and latency for easy tasks, as well as overthinking on easy tasks and under-exploration on complex tasks. Experiments on approximately 224K samples including textual and multimodal STEM tasks for the cold-start phase and ViRL39K dataset for the RLVR stage showed improved Pass@1 accuracy and Average@16 accuracy compared to baselines like GPT-4.1, Gemini-2.5-Pro-Thinking, Claude-4-Sonnet, Doubao-1.5-Thinking-Vision-Pro, Qwen2.5-VL-3B-Instruct, FAST-3B, VLAA-Thinker-3B, Qwen2.5-VL-7B-Instruct, OpenVLThinker-1.2-7B, MM-Eureka-Qwen-7B, MMR1-Math-v0, ThinkLite-7B-VL, VLAA-Thinker-7B, VL-Rethinker-7B, and Vision-G1[^7].**

**[Cheng Yang] from [Central South University and Shanghai Artificial Intelligence Laboratory] and colleagues proposed MUSE (Memory-Utilizing and Self-Evolving), an experience-driven self-evolving agent framework for long-horizon productivity tasks. The main contributions include the Memory Module (with Strategic, Procedural, and Tool Memory), Planning-Execution (PE) Agent, and Reflect Agent, forming a Plan-Execute-Reflect-Memorize loop for continuous learning and adaptation. This approach tackles the challenges faced by existing AI agents, such as their inability to continuously learn and improve over time, and their failure to effectively utilize past experiences to enhance future performance. Experiments on the TAC (TheAgentCompany) dataset showed enhancements in Partial Completion Score ($S_{partial}$), Checkpoint Score ($S_{ckpt}$), and Perfect Completion Rate (PCR) compared to baselines like Openhands, OWL-RolePlay, and Openhands-Versa[^28].**

**[Yifan Li] from [Gaoling School of Artificial Intelligence, Renmin University of China and Beijing Key Laboratory of Research on Large Models and Intelligent Governance] and colleagues introduced Perception-Time Scaling (PTS), a novel paradigm that enhances the visual perception and reasoning capabilities of Large Vision-Language Models (LVLMs). PTS reformulates perception as a structured, step-by-step reasoning process, encouraging token-rich descriptions of perceptual outcomes and decomposing complex perception tasks into simpler sub-problems. This method significantly improves the high-precision performance in visual estimation tasks such as length, perimeter, and area estimation. Experiments on DisTANCE, Geoperception, LEGO-Puzzles, MathVision, MMBench, MMVet, HalluBench, CV-Bench, and BLINK datasets showed relative accuracy improvements (RA0.1, RA${}_{\text{avg}}$) compared to baseline methods including Direct, CoT, Spatial-aware LVLMs, and Tool-augmented LVLMs[^82].**

### Technical Trends

The technical trends observed in the papers under the topic of multimodal reasoning and perception involve the development of adaptive mechanisms to enhance model performance and efficiency. ARES focuses on adapting reasoning effort based on task difficulty through entropy shaping and KL regularization, while MUSE emphasizes continuous learning and adaptation through memory utilization and self-reflection. PTS innovates by reformulating perception as a step-by-step process and integrating reinforcement learning to improve precision in visual reasoning tasks. These approaches collectively highlight the importance of adaptivity and structured learning in improving the effectiveness of multimodal reasoning models.

### Datasets and Evaluation

The primary datasets utilized across the papers include STEM tasks datasets (approx. 224K samples), ViRL39K, TAC, DisTANCE, Geoperception, LEGO-Puzzles, MathVision, MMBench, MMVet, HalluBench, CV-Bench, and BLINK. Evaluation metrics vary but generally include accuracy measures, such as Pass@1, Average@16, Relative Accuracy (RA0.1, RA${}_{\text{avg}}$), Partial Completion Score ($S_{partial}$), Checkpoint Score ($S_{ckpt}$), and Perfect Completion Rate (PCR). These datasets and metrics provide a comprehensive assessment of the multimodal reasoning and perception capabilities of the proposed models and frameworks, ensuring that their performance can be measured against established benchmarks and other state-of-the-art methods.

---

## Topic 4: Safety and Bias Mitigation

### Topic Overview

Safety and bias mitigation in large language models (LLMs) is a critical area of research aimed at ensuring these systems respond reliably and ethically across a wide range of applications. As LLMs become increasingly integrated into everyday technology, concerns about their potential to generate harmful content, exhibit biased behavior, or fail to provide helpful responses have grown. Addressing these challenges requires sophisticated methodologies for evaluating model safety and developing mechanisms to reduce risks associated with adversarial attacks, exaggerated safety behaviors, and the dissemination of misinformation.

### Individual Paper Contributions

**[Muxi Diao] from [Beijing University of Posts and Telecommunications] and colleagues proposed AutoRed, a free-form adversarial prompt generation framework for automated red teaming. With main contributions being the development of a framework that generates semantically diverse adversarial prompts without relying on seed instructions, and the use of persona data to guide prompt creation alongside an adversarial instruction verifier to improve data synthesis efficiency. Experiments on [AutoRed-Hard, AutoRed-Medium] datasets showed significant improvements in the diversity and harmfulness of generated prompts compared to [StrongR, Beaver, HarmfulQA, HarmfulQ, CodeChameleon, ReNeLLM, Jailbroken, GPTFuzzer][^11].**

**[Shuzhou Yuan] from [ScaDS.AI and TU Dresden] and colleagues introduced the Exaggerated Safety Benchmark (XSB) and Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB) along with post-hoc mitigation strategies. Their main contributions include systematic evaluations of exaggerated safety behaviors in LLMs, the identification of refusal-inducing tokens through post-hoc explanation techniques, and the implementation of lightweight, model-agnostic mitigation strategies. Experiments on [XSB, MS-XSB] datasets demonstrated improved compliance rates and reduced refusal rates compared to [No Mitigation, Ignore-Word Instruction, Prompt Rephrasing, Attention Steering][^17].**

**[Nouar Aldahoul] from [New York University Abu Dhabi] and colleagues developed a retrieval-augmented generation (RAG) approach using a multilingual LLM (RAG-Llama) to enhance the detection of adversarial misinformation attacks. The key contributions are a multi-agent system that includes a web crawler, manager, misinformation detection, topic categorization, and judge agents, as well as the use of advanced embedding models to support multilingual capabilities. Evaluations across [Dataset of 5,000 false and 2,000 factual headlines, MCQ dataset, Translation dataset, Summarization dataset] showed improvements in false and true detection accuracy, and robustness against adversarial attacks compared to [Base Llama, OpenAI’s ChatGPT 4.0 beta, Google’s Gemini, Meta’s Llama-3.1-8B][^69].**

**[Eric Hanchen Jiang] from [University of California, Los Angeles] and colleagues presented Energy-Driven Steering (EDS), a method to reduce false refusals in LLMs. The primary contributions include a fine-tuning-free framework that uses a lightweight, externally trained Energy-Based Model (EBM) to steer internal activations during inference, thereby distinguishing between desirable and undesirable outputs. Experiments on [ORB-H, XSTest-S(H), OKTest, MMLU, ARC-C, MATH, WGTest, HarmBench, WJB, DAN, X-Teaming, SafeDialBench] datasets revealed enhanced compliance rates and accuracy while maintaining low attack success rates, surpassing [System prompt, Surgical vector, CAST, AdaSteer, AlphaSteer, Defender-Only, Self-Play, Defender-Only + SFT, Self-Play + SFT][^94].**

### Technical Trends

The papers reviewed exhibit a trend toward more dynamic and nuanced approaches to safety and bias mitigation in LLMs. **AutoRed** emphasizes the importance of generating diverse adversarial prompts to thoroughly test model robustness, moving away from static datasets. **Beyond Over-Refusal** focuses on balancing safety and helpfulness by quantifying and mitigating exaggerated safety behaviors, using post-hoc explanations to understand model decisions better. **Toward a Safer Web** highlights the necessity of multilingual capabilities and advanced embedding models in combating adversarial misinformation, demonstrating the effectiveness of a multi-agent system in cross-language environments. Lastly, **Energy-Driven Steering** showcases a new direction in steering LLMs' internal activations to reduce false refusals, achieving this without requiring fine-tuning of the model.

### Datasets and Evaluation

**Main Datasets:**
- AutoRed-Hard, AutoRed-Medium
- XSB, MS-XSB
- Dataset of 5,000 false and 2,000 factual headlines, MCQ dataset, Translation dataset, Summarization dataset
- ORB-H, XSTest-S(H), OKTest, MMLU, ARC-C, MATH, WGTest, HarmBench, WJB, DAN, X-Teaming, SafeDialBench

**Evaluation Metrics:**
- Attack Success Rate (ASR)
- Compliance rates
- Refusal rates
- False detection accuracy
- True detection accuracy
- Accuracy (Acc)

These metrics are employed to assess the effectiveness of the proposed methods in mitigating safety issues and biases within LLMs, providing a comprehensive view of their performance in various contexts.

---

## Topic 5: Language Model Evaluation and Benchmarking

### Topic Overview

Language model evaluation and benchmarking are critical areas in the field of natural language processing (NLP), especially as large language models (LLMs) become increasingly sophisticated and widely applied. The primary goal of these studies is to develop reliable methods for assessing the capabilities of LLMs across various tasks, ensuring that these models generalize well to unseen data and exhibit robustness against biases and data leakage. Effective benchmarking also plays a pivotal role in guiding the development of future models, helping researchers understand where current models excel and where they fall short.

### Individual Paper Contributions

**[Qin Liu] from [University of California, Davis] and colleagues proposed ArenaBencher, a model-agnostic framework for automatic benchmark evolution.** The main contributions of ArenaBencher include its ability to aggregate feedback from diverse language models to mitigate bias and overfitting, design an ability-aware, failure-sensitive update mechanism, and develop an iterative refinement strategy using in-context demonstrations to enhance challenge and diagnostic relevance. Experiments on GSM8K (mathematical reasoning), AdvBench Harmful Behaviors (safety), and CommonsenseQA (commonsense reasoning) demonstrated improvements in accuracy, fairness, separability, and difficulty compared to the original benchmark versions and a single-model feedback variant [^3].

**[Gregory Yauney] from [University of Southern California] and colleagues introduced MDAD (Minimum Detectable Ability Difference) as a new meta-evaluation metric.** The key contributions of this paper include proposing MDAD to assess the reliability of micro-benchmarking techniques in reflecting full benchmark performance and highlighting the effectiveness of random sampling in comparison to sophisticated micro-benchmarking methods. The study evaluated 366 models from the Open LLM Leaderboard and 470 models tagged as official on the Open LLM Leaderboard v2 using datasets such as MMLU, BIG-Bench Hard (BBH), MMLU-Pro, and GPQA. It found that MDAD provided a more nuanced assessment of micro-benchmark reliability than aggregate ranking correlations, particularly in terms of agreement probability and MDAD [^56].

**[Cai Zhou] from [Massachusetts Institute of Technology] and colleagues developed the Hierarchical Diffusion Language Model (HDLM).** HDLM addresses the limitations of existing discrete diffusion models by introducing a hierarchical vocabulary and a time-varying next-scale prediction process. This innovation includes the development of a theoretical framework based on continuous-time Markov chains and the derivation of closed-form expressions for the diffusion Evidence Lower Bound (ELBO). The method was tested on the OpenWebText (OWT) dataset, showing improvements in validation and generative perplexity compared to baselines like GPT2, Llama110M, SEDD, MDLM-small, and GIDD+-small [^60].

**[V. S. Raghu Parupudi] from [University of California, San Diego] and colleagues presented a novel framework for the systematic diagnosis of brittle reasoning in large language models.** This framework automates the post-hoc diagnosis of reasoning failures and categorizes distinct reasoning modes, providing a detailed cognitive profile of the models. Key components include structured reasoning elicitation, automated diagnosis, unsupervised clustering of reasoning sentences, and quantification of reliability. Using the GSM8K dataset, the study revealed insights into the reliability of reasoning modes in gpt-3.5-turbo, text-embedding-3-large, and gpt-4o-mini, contributing to a better understanding of the brittleness in certain reasoning patterns [^75].

**[Chuyi Tan] from [Beijing Institute of Technology] and colleagues proposed Reinforcement Learning with Ensembled Rewards (RLER).** The main contributions of RLER are its use of ensemble techniques to construct a unified reward space and its implementation of adaptive soft-reward interpolation and confidence-disagreement balanced rollout selection to mitigate system bias. These innovations improve the accuracy, unbiasedness, and robustness of RL algorithms in environments with limited labeled data. Experiments on Arithmetic Dataset, DAPO-Math-17K, and Big-Math with the Qwen2.5 Series model showed improvements in test accuracy, diversity gain, and reduced reward noise rate compared to baselines such as RLVR, LLM-as-a-Judge, Self-Consistency, and Frequency-Based [^80].

### Technical Trends

The papers in this collection highlight several technical trends in the evaluation and benchmarking of large language models. Firstly, there is a shift towards more dynamic and adaptive benchmarking frameworks that evolve alongside the advancements in model capabilities, as seen in ArenaBencher. Secondly, the introduction of new meta-evaluation metrics like MDAD underscores the need for a deeper understanding of the reliability of micro-benchmarks. Thirdly, the hierarchical approach in HDLM reflects a growing interest in structured and modular solutions to enhance the flexibility and quality of language generation. Lastly, the use of ensemble techniques in RLER demonstrates an innovative way to address system bias and improve the robustness of reinforcement learning in the context of LLMs.

### Datasets and Evaluation

The papers utilize a range of datasets and evaluation metrics to assess the performance of their proposed methods. Common datasets include GSM8K for mathematical reasoning, AdvBench Harmful Behaviors for safety assessments, and CommonsenseQA for evaluating commonsense reasoning abilities. Other notable datasets include MMLU, BIG-Bench Hard (BBH), MMLU-Pro, GPQA, OpenWebText (OWT), Arithmetic Dataset, DAPO-Math-17K, and Big-Math. Evaluation metrics span from traditional accuracy and perplexity measurements to more nuanced metrics like Attack Success Rate (ASR), Fairness, Separability, Alignment, Agreement Probability, Diversity Gain, and Reward Noise Rate, reflecting the multifaceted nature of language model evaluation.

---

## Topic 6: Adaptive and Continuous Learning

### Topic Overview

Adaptive and Continuous Learning is a critical area of research within the field of artificial intelligence, particularly in the context of large language models (LLMs). These models have shown remarkable capabilities in understanding and generating human-like text, but they often struggle with adaptability to new concepts and maintaining coherence over extended interactions. This topic addresses these challenges by proposing innovative methods that allow LLMs to continuously learn and integrate new knowledge, improve their controllability, manage context dynamically, and align more closely with diverse user preferences. Enhancements in these areas are essential for developing more robust, interpretable, and versatile AI systems that can interact effectively in real-world scenarios.

### Individual Paper Contributions

**John Hewitt from Google DeepMind and colleagues proposed Neologism Learning**, with main contributions being an in-depth evaluation of how language models can be aligned with human-defined concepts through the introduction of new words (neologisms), discovering machine-only synonyms, and demonstrating the method’s applicability to both simple and complex concepts. Experiments on LIMA *(Zhou et al., [2023]) and AxBench *(Wu et al., [2025]) showed improvements in the models' ability to self-verbalize concepts and adhere to specific instructions compared to baselines such as likelihood loss training and in-context learning using example prompts [^2].

**Qiaoyu Tang from the Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences and colleagues proposed DeepMiner**, a framework for training deep search agents with dynamic context window management. Main contributions include the development of a reverse construction method for creating high-complexity QA pairs from real-world sources and a dynamic sliding window strategy that allows for sustained interactions without context limits. Experiments on HotpotQA, TriviaQA, 2WikiMultihopQA, BrowseComp, BrowseComp-zh, XBench-DeepSearch, and GAIA showed increased accuracy and extended interaction turn counts compared to commercial systems and advanced general models [^13].

**Jason Bohne from Stony Brook University and colleagues proposed Mix- and MoE-DPO**, a variational inference approach to Direct Preference Optimization (DPO) that extends standard DPO with soft mixture models and mixture-of-experts architectures. Main contributions are the ability to generalize via universal function approximation through mixtures, specialize reward and policy functions for distinct preference modes, and support contextual alignment through input-dependent soft gating. Experiments on IMDb and Amazon Book Reviews datasets demonstrated improvements in sentiment, informativeness, and grammar when compared to baseline DPO [^38].

**Shikun Liu from Georgia Institute of Technology and colleagues proposed Struc-EMB**, a structure-aware encoding method for enhancing text embeddings by integrating structural information directly into the LLM’s encoding process. Main contributions include the development of sequential concatenation (Struc-Emb-Seq) and parallel caching (Struc-Emb-Par) methods, as well as techniques for context distillation and semantic balancing to mitigate the impact of noisy structural data. Experiments on MuSiQue, HotpotQA, Cora, Citeseer, Pubmed, Books-History, Sports-Fitness, STaRK-Amazon, and StackExchangeClustering showed significant improvements in nDCG@10, Recall@k, Accuracy, Macro F1, Hit@k, MRR, and V-measure compared to individual embedding and post-hoc aggregation methods [^88].

### Technical Trends

The papers under this topic exhibit a clear trend towards enhancing the adaptive and continuous learning capabilities of LLMs through innovative methodologies. John Hewitt's team focuses on expanding the model's vocabulary and optimizing its understanding of new concepts. Qiaoyu Tang's group emphasizes the creation of complex tasks and dynamic context management for sustained interactions. Jason Bohne's team tackles the challenge of aligning LLMs with diverse user preferences using a mixture-of-experts approach. Lastly, Shikun Liu's team integrates structural information directly into the embedding process, offering a modular framework for better contextual understanding. Each approach highlights the importance of flexibility, scalability, and robustness in LLM design, indicating a shift towards more sophisticated and specialized training paradigms.

### Datasets and Evaluation Metrics

**Datasets Used:**
- LIMA *(Zhou et al., [2023])
- AxBench *(Wu et al., [2025])
- HotpotQA
- TriviaQA
- 2WikiMultihopQA
- BrowseComp
- BrowseComp-zh
- XBench-DeepSearch
- GAIA
- IMDb dataset for movie reviews
- Amazon Book Reviews dataset for book reviews
- MuSiQue
- Cora
- Citeseer
- Pubmed
- Books-History
- Sports-Fitness
- STaRK-Amazon
- StackExchangeClustering

**Evaluation Metrics:**
- Word count for long and short texts
- Sentence count for single-sentence concepts
- Prevalence of specific words
- LLM scoring for complex concepts
- AxBench evaluation scores (concept, fluency, instruct)
- Accuracy on deep research benchmarks
- Number of interaction turns
- Sentiment
- Informativeness
- Grammar
- nDCG@10
- Recall@k
- Accuracy
- Macro F1
- Hit@k
- MRR
- V-measure

These metrics collectively provide a comprehensive assessment of the models' adaptability, controllability, and contextual understanding across a variety of tasks and datasets, reflecting the multifaceted nature of adaptive and continuous learning in LLMs.

---

## Topic 7: Specialized Domain Applications

### Topic Overview
The research papers collected under the topic of Specialized Domain Applications explore the development and evaluation of large language models (LLMs) and vision-language models (VLMs) for specific tasks that require specialized reasoning abilities. These tasks include spatial reasoning, sign language understanding, long-horizon reasoning, review quality assessment, privacy analysis, culturally-grounded evaluation, and robustness against adversarial inputs. Each paper addresses unique challenges within its domain, contributing to the broader goal of enhancing the applicability and reliability of AI models in real-world scenarios.

### Individual Paper Contributions
**[Hongxing Li] from [Zhejiang University] and colleagues proposed SpatialLadder, a progressive training framework for building spatial reasoning capabilities in Vision-Language Models (VLMs). With main contributions being the introduction of a comprehensive multimodal dataset for spatial reasoning and a three-stage hierarchical training framework that progressively develops spatial intelligence from object localization to complex reasoning tasks. Experiments on [SpatialLadder-26$k$, ScanNet, SR-91k, VSI-Bench, SPBench-SI, SPBench-MV, CV-Bench, SPAR-Bench, ViewSpatial-Bench] demonstrated improved accuracy and visual attention metrics compared to [GPT-4o, Gemini-2.0-Flash, InternVL-2.5-4B, InternVL-2.5-8B, Kimi-VL-A3B, LLaVA-OneVision-7B, SpaceR-7B, VILASR-7B, Video-R1, Spatial-MLLM-4B][^29].**

**[Onur Keleş] from [Max Planck Institute for Psycholinguistics] and colleagues introduced the Visual Iconicity Challenge, a novel benchmark for evaluating Vision-Language Models (VLMs) on sign language iconicity. The main contributions include the creation of a video-based benchmark that assesses models' sensitivity to dynamic human actions in sign language and the incorporation of human baselines for phonological form and transparency. Experiments on [Sign Language of the Netherlands (NGT)] showed enhanced accuracy and correlation with human judgments on iconicity ratings compared to [Human baselines from a deaf signer and hearing sign-naïve participants][^34].**

**[Yi Lu] from [Fudan University] and colleagues developed R-Horizon, a method aimed at stimulating long-horizon reasoning behaviors in large reasoning models (LRMs). The key innovation lies in the construction of a long-horizon reasoning benchmark and the use of query composition to link problems through dependencies, thereby addressing the issue of limited effective reasoning length. Experiments on [MATH500, LiveCodeBench, WebShaper, AIME24, AIME25] demonstrated significant performance improvements on composed problems compared to [Training with single-problem data, Training with naive training data (single problems)][^41].**

**[Xiaochong Lan] from [Tsinghua University] and colleagues proposed AutoQual, an LLM agent designed for automated discovery of interpretable features for review quality assessment. The primary contribution is the integration of reflective feature search, autonomous tool implementation, and a dual-level memory architecture to bridge the gap between tacit knowledge in data and explicit, interpretable features. Experiments on [Amazon review dataset, Meituan private dataset] showed increased accuracy and reduced mean absolute error compared to [Bag-of-Words (BoW), Fixed PLM, Finetuned PLM, LLM-based methods in zero-shot and 20-shot settings, TNN, SEHP, BHeIP-CoRT][^43].**

**[Jianhui Yang] from [Tsinghua University] and colleagues presented TaoSR-AGRL, an adaptive guided reinforcement learning framework for improving search relevance in e-commerce platforms. The framework addresses reward sparsity and exploration issues through rule-aware reward shaping and adaptive guided replay. Experiments on [Balanced Eval Set (B-Eval), In-the-Wild Eval Set (W-Eval)] revealed higher F1 scores and accuracy rates on complex queries compared to [TbStar-DPO, GRPO, GRPO-PR][^44].**

**[Krzysztof Mrozinski] from [Yaraku, Inc] and colleagues focused on Quality Estimation Reranking for document-level translation. The main contribution was the exploration of QE reranking techniques to enhance the quality of document-level machine translations. Experiments on [WMT23 test set] showed improved BLEURT-20 and COMET-22 scores compared to [Standard decoding without QE reranking][^52].**

**[Weihua Zheng] from [Singapore University of Technology and Design] and colleagues introduced MMA-ASIA, a multilingual and multimodal alignment framework for culturally-grounded evaluation. The framework includes a tri-modal aligned benchmark and a five-dimensional evaluation protocol, along with the innovative Vision-ablated Prefix Replay (VPR) method to mitigate reasoning hallucinations. Experiments on [MMA-ASIA] demonstrated higher accuracy and cross-lingual consistency compared to [Existing culture-related benchmark datasets][^71].**

**[Shahriar Kabir Nahin] from [University of South Florida] and colleagues analyzed the indirect but pervasive risk of test-time scaling (TTS) in large language models (LLMs). They introduced RefDiv, a reference-guided diversity stress test protocol, to highlight vulnerabilities in TTS methods to adversarial inputs. Experiments on [AdvBench] showed successful reduction in candidate diversity and increased attack success rates compared to [GCG (Greedy Coordinate Gradient), AutoDAN][^79].**

### Technical Trends
The papers exhibit several technical trends in the development of specialized domain applications for AI models. These include:
- **Hierarchical Training and Benchmark Development**: Papers like SpatialLadder and R-Horizon emphasize the importance of structured, hierarchical training frameworks to build reasoning capabilities progressively.
- **Multimodal Integration**: There is a strong emphasis on leveraging multiple data modalities (e.g., text, images, videos) to improve model performance and generalization, as seen in SpatialLadder and MMA-ASIA.
- **Reinforcement Learning Enhancements**: TaoSR-AGRL and SpatialLadder incorporate reinforcement learning with verifiable rewards to address specific challenges in their respective domains.
- **Automatic Feature Discovery**: AutoQual showcases advancements in automatic feature discovery and interpretability, aiming to make deep learning models more transparent and adaptable.
- **Privacy and Robustness Analysis**: Less Diverse, Less Safe highlights the necessity for robustness testing against adversarial inputs, particularly in TTS methods, indicating a growing concern for the safety and reliability of AI models.

### Datasets and Evaluation Metrics
The papers utilize a variety of datasets and evaluation metrics to validate their contributions:

**Datasets**
- SpatialLadder-26$k$, ScanNet, SR-91k, VSI-Bench, SPBench-SI, SPBench-MV, CV-Bench, SPAR-Bench, ViewSpatial-Bench
- Sign Language of the Netherlands (NGT)
- MATH500, LiveCodeBench, WebShaper, AIME24, AIME25
- Amazon review dataset, Meituan private dataset
- Balanced Eval Set (B-Eval), In-the-Wild Eval Set (W-Eval)
- WMT23 test set
- MMA-ASIA
- AdvBench

**Evaluation Metrics**
- Accuracy, Visual Attention IoU, Visual Attention Entropy, Semantic Entropy
- Accuracy on phonological form prediction, Transparency task accuracy, Correlation with human iconicity judgment (Spearman's rho), Separation of iconic vs. arbitrary signs (Cohen's d)
- Accuracy on composed problems, Expected accuracy, Performance improvement on original tasks
- Spearman’s Rho ($r_{s}$), Mean Absolute Error (MAE), F1-Score, Area Under the ROC Curve (AUROC)
- Class-1 F1, Class-2 F1, Class-3 F1, Class-4 F1, Good F1, Macro F1, Accuracy, Good/Same/Bad (GSB), Query Goodrate, Item Goodrate
- BLEURT-20, COMET-22, GEMBA-DA
- Accuracy, Cross-lingual consistency, Cross-modal consistency, Cultural knowledge generalization, Grounding validation
- Attack Success Rate (ASR)

These datasets and metrics reflect the diversity of the tasks and the importance of domain-specific evaluations in ensuring that AI models perform reliably and accurately in specialized applications.

---

## Topic 8: Sentiment and Emotion Analysis

**Topic Overview**: Sentiment and emotion analysis is a critical area in natural language processing (NLP) that involves understanding and extracting opinions, attitudes, and emotions from textual data. This field is increasingly important as it helps in gauging public perception and reaction towards various events, products, or services, especially during times of crisis such as the global COVID-19 pandemic. Additionally, the integration of multiple modalities (text, audio, video) enhances the capability to recognize complex emotional states in human interactions, which is essential for applications ranging from customer service to mental health support.

**Individual Paper Contributions**:

**[Qiang Yang] from [King Abdullah University of Science and Technology] and colleagues proposed SenWave, a fine-grained multi-language sentiment analysis dataset sourced from COVID-19 tweets. With main contributions being the introduction of a comprehensive annotated dataset for fine-grained sentiment analysis and the development of specialized sentiment labels to capture the nuanced public emotions during a health crisis. Experiments on [SenWave: 105 million unlabeled tweets; 10,000 labeled tweets each in English and Arabic] showed significant improvements in accuracy and F1 scores across various sentiment categories compared to baseline methods such as FastText, CNN, LSTM, and other BERT-based models[^16].**

**[Yu Liu] and colleagues introduced the Centering ERC on emotion hotspots through Hotspot-Gated Fusion (HGF) and Mixture-of-Aligners (MoA) for cross-modal alignment in the paper titled "Centering Emotion Hotspots". The main contributions were identifying and weighting localized high-intensity segments within each modality, performing flexible cross-modal alignment, and modeling relational structures in conversations. Experiments on [IEMOCAP, CMU-MOSEI] demonstrated enhanced accuracy and weighted F1-scores compared to DialogueRNN, DialogueGCN, MMGCN, COGMEN, and other baselines[^68].**

**[Deshui Yu] from [Tsinghua University Shenzhen International Graduate School] and colleagues developed the YpathRAG framework, a retrieval-augmented generation model specifically for the pathology domain. Main contributions included constructing a large-scale pathology vector database, designing pathology-specific evaluation benchmarks (YpathR and YpathQA-M), integrating dense and sparse retrieval mechanisms, and implementing an LLM-based support judgment module. Experiments on [YpathR, YpathQA-M] showed improvements in precision, coverage, faithfulness, and semantic similarity compared to general-purpose models like BGE-M3, Qwen3-Embedding-8B, and Qwen3-Ranker-8B[^73].**

**Technical Trends**: The papers reflect a shift towards more specialized and fine-grained approaches in sentiment and emotion analysis. SenWave emphasizes the creation of rich, multi-language datasets with detailed annotations to address the specific needs of sentiment analysis during health crises. The second paper by Yu Liu et al. highlights advancements in multimodal fusion techniques, particularly focusing on the identification of emotion-relevant segments and their integration with global context. Lastly, Deshui Yu's work showcases the importance of domain-specific knowledge integration, utilizing retrieval-augmented generation frameworks to enhance the reliability and accuracy of large language models in specialized fields like pathology.

**Datasets and Evaluation**: 
- **SenWave**: Utilizes a massive collection of 105 million unlabeled tweets and 10,000 labeled tweets each in English and Arabic. Evaluation metrics include accuracy, F1-macro, F1-micro, ranking average precision score (LRAP), and Hamming loss.
- **IEMOCAP and CMU-MOSEI**: These datasets are used for evaluating the ERC system with Hotspot-Gated Fusion (HGF) and Mixture-of-Aligners (MoA). Metrics include accuracy and weighted F1-score.
- **YpathR and YpathQA-M**: These are pathology-specific evaluation benchmarks designed to assess the performance of the YpathRAG framework. Metrics evaluated are Precision@5, Hit@6, MeanRank, IOR-Global, IOR-Positive, Keyword, Coverage, Faithfulness, and Semantic Similarity. 

These studies collectively advance the methodologies for sentiment and emotion analysis, addressing specific challenges such as multi-modality integration, fine-grained labeling, and domain specificity, thus enriching the field with new tools and approaches for analyzing complex human sentiments and emotions.

---

## Topic 9: Automated Code and Test Case Generation

### Topic Overview

Automated code and test case generation is an essential area of research in software engineering and artificial intelligence. The goal is to develop algorithms and frameworks that can automatically produce high-quality code and corresponding test cases based on specifications or examples, significantly reducing human effort and improving software development efficiency. This topic is increasingly important due to the growing complexity of software systems and the need for rapid development cycles while maintaining high standards of quality and reliability.

### Individual Paper Contributions

**Tengxiao Lv from [Institution] and colleagues proposed a Unified Biomedical Named Entity Recognition Framework with Large Language Models, with main contributions being the introduction of a symbolic tagging strategy for entity boundary representation, multi-dataset joint fine-tuning for cross-lingual generalization, and a contrastive entity selector to enhance precision. Experiments on CMeEE-V2 (Chinese), CCKS2019-AS (Chinese), GENIA (English), BioRED (English), BC5CDR-Chemical (English), and NCBI-Disease (English) showed significant improvements in Precision (P), Recall (R), and F1 Score (F1) compared to BERT-based and other LLM-based approaches[^50].**

**Suming Qiu from [Institution] and colleagues proposed HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance, with main contributions being a hybrid training framework that integrates self-distillation with reinforcement learning (RL) fine-tuning, a skeleton-completeness scoring mechanism, and a query-latency-aware reward system. These innovations aim to improve the accuracy and efficiency of SQL query generation from natural language inputs, addressing the challenges of complex reasoning and compositional generalization. Experiments on BIRD, Spider, and KaggleDBQA datasets showed enhanced Exact-setmatch Accuracy (EM), Execution accuracy (EX), Valid Efficiency Score (VES), Performance Gap Recovered (PGR), and Token Elasticity of Performance (TEP) compared to various baselines[^81].**

**Tinnakit Udsa from School of Information Science and Technology, VISTEC and colleagues proposed a framework to measure intra- and inter-client memorization of training data in federated learning (FL) settings, with main contributions being the adaptation of cross-sample memorization assessment from centralized learning (CL) to FL and the development of memorization metrics including the pairwise memorization ratio ($\mathcal{\sf MR}_{j\rightarrow k}$). This work addresses the critical issue of privacy preservation in FL, especially in sensitive domains like healthcare, by providing a realistic assessment of memorization risks. Experiments on summarization, dialog, QA, and classification tasks using Qwen2.5-3B and Llama3.2 models showed a more comprehensive evaluation of memorization compared to existing methods like Canary Injection and BLEU Score[^89].**

### Technical Trends

The papers collectively highlight several key trends in automated code and test case generation:
1. **Use of Large Language Models (LLMs)**: All three papers leverage LLMs for their respective tasks, indicating a trend towards utilizing advanced AI models for complex and domain-specific text processing.
2. **Fine-Tuning Techniques**: The application of fine-tuning on diverse datasets, either through joint fine-tuning or hybrid methods involving self-distillation and RL, showcases an evolving methodology aimed at enhancing model performance and generalization capabilities.
3. **Innovative Metric Development**: Each paper introduces novel metrics to assess model performance, such as $\mathcal{\sf MR}_{j\rightarrow k}$ for memorization in federated learning, Exact-setmatch Accuracy for SQL generation, and Precision, Recall, and F1 Score for named entity recognition. This reflects a continuous effort to refine and improve evaluation methods in the field.

### Datasets and Evaluation

**Main Datasets Used:**
- CMeEE-V2, CCKS2019-AS, GENIA, BioRED, BC5CDR-Chemical, NCBI-Disease (for BioNER)
- BIRD, Spider, KaggleDBQA (for Text-to-SQL)
- Summarization, Dialog, Question Answering (QA), Classification (for federated learning memorization)

**Evaluation Metrics:**
- Precision (P), Recall (R), F1 Score (F1) (for BioNER)
- Exact-setmatch Accuracy (EM), Execution accuracy (EX), Valid Efficiency Score (VES), Performance Gap Recovered (PGR), Token Elasticity of Performance (TEP) (for Text-to-SQL)
- $\mathcal{\sf MR_{Intra}}$, $\mathcal{\sf MR_{Inter}}$, $\mathcal{\sf MR_{TotalCL}}$, $\mathcal{\sf MR_{TotalFL}}$ (for federated learning memorization)

These metrics are carefully chosen to evaluate the specific aspects of performance relevant to each task, reflecting the diversity and complexity of the research questions addressed in the field of automated code and test case generation.

---

## Topic 10: Miscellaneous Topics

### Topic Overview

This collection of research papers delves into a variety of miscellaneous topics related to the development, optimization, and evaluation of large language models (LLMs) and their applications across different domains. The importance of this research lies in its efforts to uncover and address various limitations and biases present in LLMs, improve their performance in specialized tasks, and enhance their applicability in real-world scenarios. Each paper brings unique insights into specific challenges, such as cultural understanding, judgment preference bias, and reasoning capabilities, and proposes innovative methods to tackle these issues.

### Individual Paper Contributions

**[Ioana Marinescu] from [NYU] and colleagues proposed an optimization algorithm to enumerate a spectrum of possible label sets varying in semantic relevance, with main contributions being disentangling the roles of representation and learning in in-context learning (ICL) performance, and a systematic method to find optimal label sets for ICL tasks. Experiments on sentiment classification datasets showed that the two aspects are largely orthogonal and that representation sets a baseline while learning improves upon it[^8].**

**[Taisei Yamamoto] from [The University of Tokyo] and colleagues proposed the Culture Neuron Identification Pipeline with Gradient-based Scoring (CULNIG), with main contributions being the identification of culture-general and culture-specific neurons, and the use of gradient-based scoring for neuron selection. Experiments on BLEnD, CulturalBench, NormAd, WorldValuesBench, CommonSenseQA, QNLI, MRPC, and CRC datasets demonstrated a precise identification of neurons responsible for cultural understanding, distinguishing between general and specific cultural knowledge[^12].**

**[Tim Hagen] from [University of Kassel] and colleagues proposed the Concausal News Corpus (CCNC) and extending the causality extraction task to include concausal statements, with main contributions being the identification and inclusion of concausal statements in causality extraction datasets and the creation of a new dataset (CCNC) for training and evaluation of models on procausal and concausal claims. Experiments on CCNC and CNCv2 datasets showed improved accuracy and balanced causal reasoning compared to UniCausal baseline[^14].**

**[Jian Xie] from [The Ohio State University] and colleagues proposed ARM2, an adaptive reasoning model with vision understanding and executable code, with main contributions being the incorporation of executable code into adaptive reasoning, GRPO-alp which explicitly incorporates response length into the optimization objective, and support for multimodal reasoning including vision understanding. Experiments on AQuA-Rat, VisualWebInstruct, CommonsenseQA, GSM8K, AIME, Geometry3K, MME RealWorld, MMK12, OBQA, Math500, GPQADiamond, BLINK, ChartQA, and MMMU datasets demonstrated enhanced efficiency and effectiveness of LRMs across diverse tasks[^18].**

**[Yuzheng Cai] from [No Institution Listed] and colleagues proposed Training-Free Group Relative Policy Optimization (Training-Free GRPO), with main contributions being a new training-free RL paradigm shifting policy optimization from the parameter space to the context space, semantic group advantage replacing numerical group advantage for policy optimization, and data and computational efficiency with minimal training samples. Experiments on DAPO-Math-17K, AIME 2024 and 2025 benchmarks, AFM, and WebWalkerQA benchmark datasets showed improved performance and reduced computational costs compared to models trained using SFT and Ada-GRPO[^20].**

**[Shuliang Liu] from [Northeastern University, China] and colleagues proposed Group-Based Polling Optimization (Genii), with main contributions being an unsupervised multi-agent collaborative optimization framework for LLM-based evaluators, simulated client-server polling mechanism to enhance unbiased evaluations, and exploitation of weaker models to improve stronger models bidirectionally. Experiments on Evol-Instruct, UltraFeedback, MTbench, AutoJ, Preferencebench, Rewardbench, NQ, TriviaQA, and HotpotQA datasets demonstrated a reduction in harmful self-preference propensity and increased accuracy in judgment compared to vanilla LLMs and Self-Consistency[^22].**

**[Haoyang Gui] from [Utrecht University, The Netherlands] and colleagues proposed using LLMs to classify influencer marketing posts and generate legal reasoning explanations, with main contributions being the development of a taxonomy of common errors in LLM-generated legal reasoning specific to influencer marketing detection, original dataset of LLM explanations annotated by students trained in influencer marketing law, and a combination of quantitative and qualitative evaluation strategies for LLM explanations. Experiments on a dataset of 1,143 Instagram posts showed improvements in precision, recall, and F1 score compared to logistic regression (TF-IDF)[^23].**

**[Jasmina Gajcin] from [IBM Research, Ireland] and colleagues proposed CLoVE (Contrastive Local Verifiable Explanations) and GloVE (Global Verifiable Explanations) for generating and summarizing local and global explanations of LLM-as-a-Judge policies, with main contributions being the development of CLoVE for generating verifiable concept-based rationales for local explanations, and the introduction of GloVE for summarizing local rules into a faithful, interpretable global policy. Experiments on BeaverTails, XSTest, OpenAIMod, SafeRLHF, AgentHarm, HarmBench, and SimpleSafetyTests datasets showed improvements in policy transparency and fidelity compared to GELPE[^24].**

**[Shule Lu] from [No Institution Listed] and colleagues proposed FedDTRE, a federated dialogue generation model powered by trustworthiness evaluation, with main contributions being the introduction of trustworthiness scores to guide the integration of global and local knowledge, and an adaptive aggregation strategy based on trustworthiness evaluation to balance privacy and performance. Experiments on Synthetic-Persona-Chat, CMU_DoG, and Wizard of Wikipedia datasets demonstrated improvements in BLEU, ROUGE, and BERTScore compared to FedAvg and FedProx[^25].**

**[Jiayun Luo] from [University of British Columbia] and colleagues proposed DIYSink, a framework designed to enhance Large Vision Language Models (LVLMs) performance by optimizing the use of visual attention sinks, with main contributions being the identification and analysis of the role of ViT sinks in LVLMs, and the development of a framework to selectively emphasize or de-emphasize ViT sink tokens based on task requirements. Experiments on LLaVA eval, GQA, ScienceQA, TextVQA, MMU, MME, and MathVista datasets showed improvements in accuracy and task-specific performance metrics compared to original LVLM baselines[^32].**

**[Bianca-Mihaela Ganescu] from [ALTA Institute] and colleagues proposed a token-wise dynamic gating mechanism for adaptive fusion of linguistic and visual cues in a dual-stream transformer architecture, with main contributions being the implementation of token-wise dynamic gating for adaptive fusion of multimodal information, and the use of modulation techniques and channel attention to maximize utility of limited visual information. Experiments on BLiMP, BLiMP Supplement, EWoK, Winoground, and VQA datasets demonstrated improved performance compared to Flamingo and GIT baselines[^33].**

**[Cheng Qian] from [Salesforce AI Research] and colleagues proposed xRouter, a training cost-aware LLM orchestration system via reinforcement learning, with main contributions being the design of a tool-calling based routing system that can answer directly or delegate to external models, and demonstration of learning routing behavior with explicit cost-performance trade-offs. Experiments on Reasoning360 benchmark, AIME, Math-500, Olympiad Bench, AMC-23, Codeforces, Code-Contests, Human-EvalPlus, LiveCodeBenchv5, GPQADiamond, AIME25, MTBench, and IFEval datasets showed improvements in accuracy and cost utility compared to single-model baselines[^35].**

**[Marius Dragoi] from [Bitdefender, Romania] and colleagues proposed the Cover@$\tau$ metric, with main contributions being introducing Cover@$\tau$ to measure reasoning under an explicit reliability threshold, demonstrating that Pass@k is biased towards low-$\tau$ regions of Cover@$\tau$, and illustrating the usefulness of Cover@$\tau$ in evaluating different RLVR methods. Experiments on OMEGA and Reasoning Gym datasets showed a more nuanced view of reasoning capabilities compared to traditional Pass@k metric[^36].**

**[Marta Emili Garcia Segura] from [University College London] and colleagues proposed ShapeLLM, a model-free opponent shaping algorithm tailored for transformer-based agents, with main contributions being the first investigation of opponent shaping in LLM agents, and demonstrating that LLM agents can guide interactions towards mutually beneficial outcomes in cooperative settings and exploit opponents in competitive ones. Experiments on Iterated Prisoner’s Dilemma, Iterated Matching Pennies, Iterated Chicken Game, and Iterated Stag Hunt datasets showed improvements in cumulative trial return and successful exploitation/exploitation avoidance compared to independent LLM agents trained using PPO[^37].**

**[Daniel Huwiler] from [Zurich University of Applied Sciences, Switzerland] and colleagues proposed VersionRAG, with main contributions being the explicit modeling of version relationships and changes through a hierarchical graph, the query-aware retrieval strategy that distinguishes between content, version listing, and change retrieval, and the efficient indexing and retrieval through graph edges rather than full-document LLM analysis. Experiments on VersionQA, a manually curated dataset, showed improvements in accuracy compared to naive RAG and GraphRAG[^39].**

**[Lan Zhang] from [University of Manchester, United Kingdom] and colleagues proposed MASA, a framework for building modular agents for autoformalization leveraging the interaction between LLMs and theorem provers, with main contributions being a modular framework for building multi-agent systems for autoformalization offering flexibility and extensibility, and evaluation in multiple multi-agent settings showcasing effectiveness and valuable insights. Experiments on miniF2F and ProofNet datasets demonstrated improvements in BLEU-4, ChrF, and RUBY scores compared to zero-shot and few-shot prompting[^45].**

**[Shramay Palta] from [University of Maryland, College Park, USA] and colleagues proposed an investigation into the impact of LLM-generated rationales on human notions of plausibility, with main contributions being the collection of human and LLM plausibility ratings of answers with and without LLM-generated rationales, and the analysis of the differential impact of PRO and CON rationales on gold-standard versus distractor answers. Experiments on SIQA and CQA datasets showed significant changes in human plausibility judgments across different rationale conditions compared to no rationale condition[^26].**

**[Jiaming Wang] from [Meituan M17] and colleagues proposed SOP-Maze, a benchmark for evaluating LLMs on complex business SOP scenarios, with main contributions being the first benchmark specifically tailored for evaluating LLMs in complex business SOP scenarios, and the categorization of SOP tasks into LRS and HRS to assess different aspects of model capabilities. Experiments on SOP-Maze showed improvements in procedural compliance and robustness under noisy conditions compared to various API-based and open-source models[^48].**

**[Haomin Zhuang] from [University of Notre Dame] and colleagues proposed multi-temperature strategies for token- and rollout-level control in RLVR, with main contributions being differentiated temperature control during token sampling based on token type, and multi-temperature sampling per prompt to enhance exploration. Experiments on MATH, DAPO, AIME24, AIME25, Minerva, and Olympiad datasets demonstrated enhanced exploration and exploitation balance, leading to better reasoning performance of LLMs compared to base model and DAPO-T1.0[^51].**

**[Siddeshwar Raghavan] from [Argonne National Laboratory] and colleagues proposed MOSAIC, a training-free, multi-agent LLM framework for task-intelligent scientific coding, with main contributions being the integration of agents that collaborate to enhance code accuracy and executability without needing test cases, and the implementation of a consolidated context window to mitigate hallucinations and maintain context. Experiments on SciCode, MBPP, HumanEval, and APPS datasets demonstrated higher main problem-solving rates and subproblem-solving rates compared to direct synthesis and chain-of-thought prompting[^53].**

**[Wangjie You] from [Douyin Content Group, ByteDance] and colleagues proposed the Chinese Commonsense Multi-Hop Reasoning (CCMOR) benchmark, with main contributions being the introduction of CCMOR for evaluating multi-step reasoning capabilities of LLMs in Chinese, and the use of an LLM-powered pipeline for generating multi-hop questions with human-in-the-loop verification. Experiments on Chinese SimpleQA, CHARM-Memorization, and CCMOR datasets showed improvements in Rouge-L Recall and LLM-as-Judge Accuracy compared to System-1 and System-2 style models[^55].**

**[Li Zhang] from [University of Pittsburgh] and colleagues proposed a decomposed framework for hierarchical legal reasoning, with main contributions being the operationalization of CATO-style hierarchy into three decomposed tasks, and the empirical study revealing limitations in LLMs for complex legal reasoning. Experiments on various legal datasets showed improvements in accuracy and token usage compared to non-thinking models[^74].**

**[Xianzhen Luo] from [Harbin Institute of Technology] and colleagues proposed a benchmarking framework for code LLMs, with main contributions being the first large-scale empirical study of scaling laws for code LLMs, and the demonstration that the Farseer law better fits the scaling behavior of code LLMs compared to the Chinchilla law. Experiments on Opencoder code pre-training data and an internal codebase from Xuyang et al. demonstrated a higher optimal data-to-parameter ratio for code LLMs compared to natural language LLMs[^95].**

### Technical Trends

The papers in this collection adopt a range of technical approaches, reflecting the evolving methodologies in the field of LLM research. Key trends include:
- **Fine-Tuning and Reinforcement Learning**: Several papers utilize fine-tuning and reinforcement learning techniques to optimize LLM performance in specialized tasks, such as ARM2 and Training-Free GRPO.
- **Modular and Multi-Agent Systems**: Innovations in modular frameworks and multi-agent systems are prominent, exemplified by MOSAIC, MASA, and FedDTRE.
- **Semantic and Contextual Analysis**: Papers like CLoVE/GloVE and Text2Stories focus on enhancing the transparency and interpretability of LLM decisions through semantic and contextual analysis.
- **Prompting and Rationale Generation**: Studies like JAI-1 and Sentiment Matters explore the impact of prompting and rationale generation on model performance and bias mitigation.
- **Scaling Laws and Efficiency Improvements**: Works like Scaling Laws for Code and Recover-LoRA highlight the importance of understanding and optimizing scaling laws and efficiency in LLM training and deployment.

### Datasets and Evaluation

The papers employ a wide array of datasets and evaluation metrics to assess the performance of LLMs and related models. Some of the notable datasets include:
- Sentiment classification datasets
- BLEnD, CulturalBench, NormAd, WorldValuesBench, CommonSenseQA, QNLI, MRPC, CRC datasets
- Concausal News Corpus (CCNC) and Causal News Corpus v2 (CNCv2)
- AQuA-Rat, VisualWebInstruct, CommonsenseQA, GSM8K, AIME, Geometry3K, MME RealWorld, MMK12, OBQA, Math500, GPQADiamond, BLINK, ChartQA, MMMU datasets
- DAPO-Math-17K, AIME 2024 and 2025 benchmarks, AFM, and WebWalkerQA benchmark datasets
- VersionQA, GutBrainIE challenge dataset, miniF2F, ProofNet, and MMLU-Pro datasets
- SIQA and CQA datasets
- SOP-Maze, SciCode, MBPP, HumanEval, and APPS datasets
- LibriSpeech, MuSE, StressID, and custom real-world datasets
- VideoNorms, M-TRACE, and Pref-X datasets

Evaluation metrics used across the papers include:
- Accuracy
- Zero-shot accuracy
- N-shot accuracy
- Macro-averaged F1 score
- Precision (P)
- Recall (R)
- Mean@32, Pass@1, Pass@3
- Mean plausibility ratings
- BLEU, COMET, DA-BERT, MQM scores
- Navigation Error (NE), Oracle Success Rate (OS), Success Rate (SR)
- Micro-averaged F1-score, Macro-averaged F1-score
- HR@1
- Word Error Rate (WER)
- Accuracy Recovery Percentage (AR%)
- Three-tier scoring system
- Temporal Accuracy (Eb, At), Audio Quality (FAD, KL, FD, IS, CLAP), Speech Intelligibility (WER, MOS), Real-time Factor (RTF)
- Time to First Token, Response Time, Code Comments, Preambles and Postambles
- Step-by-Step correctness, Deep Reasoning coherence and factual accuracy, Outcome overall task-solving success, AnsAcc (Answer Accuracy)

Each paper uniquely contributes to the advancement of LLM research by addressing specific challenges and proposing innovative solutions that aim to improve the reliability, efficiency, and applicability of these models in diverse domains.

---

## References

[^0]: [Which Heads Matter for Reasoning? RL-Guided KV Cache Compression](https://arxiv.org/abs/2510.08525)

[^1]: [Efficient Prompt Optimisation for Legal Text Classification with Proxy Prompt Evaluator](https://arxiv.org/abs/2510.08524)

[^2]: [Neologism Learning for Controllability and Self-Verbalization](https://arxiv.org/abs/2510.08506)

[^3]: [ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation](https://arxiv.org/abs/2510.08569)

[^4]: [DeepPrune: Parallel Scaling without Inter-trace Redundancy](https://arxiv.org/abs/2510.08483)

[^5]: [Single layer tiny Co$^4$ outpaces GPT-2 and GPT-BERT](https://arxiv.org/abs/2510.08404)

[^6]: [LeWiDi-2025 at NLPerspectives: The Third Edition of the Learning with Disagreements Shared Task](https://arxiv.org/abs/2510.08460)

[^7]: [ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping](https://arxiv.org/abs/2510.08457)

[^8]: [On the Relationship Between the Choice of Representation and In-Context Learning](https://arxiv.org/abs/2510.08372)

[^9]: [If Probable, Then Acceptable? Understanding Conditional Acceptability Judgments in Large Language Models](https://arxiv.org/abs/2510.08388)

[^10]: [Two-Stage Voting for Robust and Efficient Suicide Risk Detection on Social Media](https://arxiv.org/abs/2510.08365)

[^11]: [AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming](https://arxiv.org/abs/2510.08329)

[^12]: [Neuron-Level Analysis of Cultural Understanding in Large Language Models](https://arxiv.org/abs/2510.08284)

[^13]: [Beyond Turn Limits: Training Deep Search Agents with Dynamic Context Window](https://arxiv.org/abs/2510.08276)

[^14]: [Investigating Counterclaims in Causality Extraction from Text](https://arxiv.org/abs/2510.08224)

[^15]: [The Alignment Waltz: Jointly Training Agents to Collaborate for Safety](https://arxiv.org/abs/2510.08240)

[^16]: [SenWave: A Fine-Grained Multi-Language Sentiment Analysis Dataset Sourced from COVID-19 Tweets](https://arxiv.org/abs/2510.08214)

[^17]: [Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs](https://arxiv.org/abs/2510.08158)

[^18]: [ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code](https://arxiv.org/abs/2510.08163)

[^19]: [DACIP-RC: Domain Adaptive Continual Instruction Pre-Training via Reading Comprehension on Business Conversations](https://arxiv.org/abs/2510.08152)

[^20]: [Training-Free Group Relative Policy Optimization](https://arxiv.org/abs/2510.08191)

[^21]: [AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents](https://arxiv.org/abs/2510.08149)

[^22]: [Mitigating Judgment Preference Bias in Large Language Models through Group-Based Polling](https://arxiv.org/abs/2510.08145)

[^23]: [Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing](https://arxiv.org/abs/2510.08111)

[^24]: [Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations](https://arxiv.org/abs/2510.08120)

[^25]: [FedDTRE: Federated Dialogue Generation Models Powered by Trustworthiness Evaluation](https://arxiv.org/abs/2510.08058)

[^26]: [Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility](https://arxiv.org/abs/2510.08091)

[^27]: [ChatGPT as a Translation Engine: A Case Study on Japanese-English](https://arxiv.org/abs/2510.08042)

[^28]: [Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks](https://arxiv.org/abs/2510.08002)

[^29]: [SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2510.08531)

[^30]: [SliceFine: The Universal Winning-Slice Hypothesis for Pretrained Networks](https://arxiv.org/abs/2510.08513)

[^31]: [AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents](https://arxiv.org/abs/2510.08511)

[^32]: [To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models](https://arxiv.org/abs/2510.08510)

[^33]: [Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling](https://arxiv.org/abs/2510.08470)

[^34]: [The Visual Iconicity Challenge: Evaluating Vision-Language Models on Sign Language Form-Meaning Mapping](https://arxiv.org/abs/2510.08482)

[^35]: [xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning](https://arxiv.org/abs/2510.08439)

[^36]: [Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries](https://arxiv.org/abs/2510.08325)

[^37]: [Opponent Shaping in LLM Agents](https://arxiv.org/abs/2510.08255)

[^38]: [Mix- and MoE-DPO: A Variational Inference Approach to Direct Preference Optimization](https://arxiv.org/abs/2510.08256)

[^39]: [VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents](https://arxiv.org/abs/2510.08109)

[^40]: [Sentiment Matters: An Analysis of 200 Human-SAV Interactions](https://arxiv.org/abs/2510.08202)

[^41]: [R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](https://arxiv.org/abs/2510.08189)

[^42]: [NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions](https://arxiv.org/abs/2510.08173)

[^43]: [AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessment](https://arxiv.org/abs/2510.08081)

[^44]: [TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance](https://arxiv.org/abs/2510.08048)

[^45]: [MASA: LLM-Driven Multi-Agent Systems for Autoformalization](https://arxiv.org/abs/2510.08988)

[^46]: [A Human Behavioral Baseline for Collective Governance in Software Projects](https://arxiv.org/abs/2510.08956)

[^47]: [Artificial Impressions: Evaluating Large Language Model Behavior Through the Lens of Trait Impressions](https://arxiv.org/abs/2510.08915)

[^48]: [SOP-Maze: Evaluating Large Language Models on Complicated Business Standard Operating Procedures](https://arxiv.org/abs/2510.08942)

[^49]: [Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors](https://arxiv.org/abs/2510.08907)

[^50]: [A Unified Biomedical Named Entity Recognition Framework with Large Language Models](https://arxiv.org/abs/2510.08902)

[^51]: [Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR](https://arxiv.org/abs/2510.08892)

[^52]: [Quality Estimation Reranking for Document-Level Translation](https://arxiv.org/abs/2510.08870)

[^53]: [MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding](https://arxiv.org/abs/2510.08804)

[^54]: [The Model's Language Matters: A Comparative Privacy Analysis of LLMs](https://arxiv.org/abs/2510.08813)

[^55]: [Benchmarking Chinese Commonsense Reasoning with a Multi-hop Reasoning Perspective](https://arxiv.org/abs/2510.08800)

[^56]: [How Reliable is Language Model Micro-Benchmarking?](https://arxiv.org/abs/2510.08730)

[^57]: [Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning](https://arxiv.org/abs/2510.08710)

[^58]: [How Many Code and Test Cases Are Enough? Evaluating Test Cases Generation from a Binary-Matrix Perspective](https://arxiv.org/abs/2510.08720)

[^59]: [Scaling Laws for Code: A More Data-Hungry Regime](https://arxiv.org/abs/2510.08702)

[^60]: [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632)

[^61]: [From What to Why: Thought-Space Recommendation with Small Language Models](https://arxiv.org/abs/2510.08626)

[^62]: [PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction](https://arxiv.org/abs/2510.08623)

[^63]: [Do LLMs Know They Are Being Tested? Evaluation Awareness and Incentive-Sensitive Failures in GPT-OSS-20B](https://arxiv.org/abs/2510.08624)

[^64]: [From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents](https://arxiv.org/abs/2510.08621)

[^65]: [Text2Stories: Evaluating the Alignment Between Stakeholder Interviews and Generated User Stories](https://arxiv.org/abs/2510.08622)

[^66]: [LLMs Show Surface-Form Brittleness Under Paraphrase Stress Tests](https://arxiv.org/abs/2510.08616)

[^67]: [JAI-1: A Thai-Centric Large Language Model](https://arxiv.org/abs/2510.08620)

[^68]: [Centering Emotion Hotspots: Multimodal Local-Global Fusion and Cross-Modal Alignment for Emotion Recognition in Conversations](https://arxiv.org/abs/2510.08606)

[^69]: [Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks](https://arxiv.org/abs/2510.08605)

[^70]: [Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection](https://arxiv.org/abs/2510.08602)

[^71]: [MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally-Grounded Evaluation](https://arxiv.org/abs/2510.08608)

[^72]: [Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs](https://arxiv.org/abs/2510.08601)

[^73]: [YpathRAG:A Retrieval-Augmented Generation Framework and Benchmark for Pathology](https://arxiv.org/abs/2510.08603)

[^74]: [Confidence, Not Perplexity: A Better Metric for the Creative Era of LLMs](https://arxiv.org/abs/2510.08596)

[^75]: [Systematic Diagnosis of Brittle Reasoning in Large Language Models](https://arxiv.org/abs/2510.08595)

[^76]: [Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation](https://arxiv.org/abs/2510.08600)

[^77]: [Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech](https://arxiv.org/abs/2510.08593)

[^78]: [Enhancing Biomedical Named Entity Recognition using GLiNER-BioMed with Targeted Dictionary-Based Post-processing for BioASQ 2025 task 6](https://arxiv.org/abs/2510.08588)

[^79]: [Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](https://arxiv.org/abs/2510.08592)

[^80]: [Diagnosing and Mitigating System Bias in Self-Rewarding RL](https://arxiv.org/abs/2510.08977)

[^81]: [HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance](https://arxiv.org/abs/2510.08896)

[^82]: [Unleashing Perception-Time Scaling to Multimodal Reasoning Models](https://arxiv.org/abs/2510.08964)

[^83]: [ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](https://arxiv.org/abs/2510.08878)

[^84]: [ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review](https://arxiv.org/abs/2510.08867)

[^85]: [Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training](https://arxiv.org/abs/2510.08855)

[^86]: [Everyone prefers human writers, including AI](https://arxiv.org/abs/2510.08831)

[^87]: [McMining: Automated Discovery of Misconceptions in Student Code](https://arxiv.org/abs/2510.08827)

[^88]: [Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings](https://arxiv.org/abs/2510.08774)

[^89]: [Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](https://arxiv.org/abs/2510.08750)

[^90]: [When to Reason: Semantic Router for vLLM](https://arxiv.org/abs/2510.08731)

[^91]: [Optimizing delivery for quick commerce factoring qualitative assessment of generated routes](https://arxiv.org/abs/2510.08671)

[^92]: [BaldWhisper: Faster Whisper with Head Shearing and Layer Merging](https://arxiv.org/abs/2510.08599)

[^93]: [Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech](https://arxiv.org/abs/2510.08586)

[^94]: [Energy-Driven Steering: Reducing False Refusals in Large Language Models](https://arxiv.org/abs/2510.08646)

[^95]: [Articulation-Informed ASR: Integrating Articulatory Features into ASR via Auxiliary Speech Inversion and Cross-Attention Fusion](https://arxiv.org/abs/2510.08585)

[^96]: [Comparative Analysis of Large Language Models for the Machine-Assisted Resolution of User Intentions](https://arxiv.org/abs/2510.08576)

[^97]: [VideoNorms: Benchmarking Cultural Awareness of Video Language Models](https://arxiv.org/abs/2510.08543)

[^98]: [MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning](https://arxiv.org/abs/2510.08567)


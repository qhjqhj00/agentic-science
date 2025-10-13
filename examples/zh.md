# 研究论文主题分析报告 - 2025年10月13日
## 目录
- [Topic 1: Memory Efficiency and Compression](#topic-1-memory-efficiency-and-compression) (3 papers)
- [Topic 2: Legal and Regulatory Applications](#topic-2-legal-and-regulatory-applications) (3 papers)
- [Topic 3: Multimodal Reasoning and Understanding](#topic-3-multimodal-reasoning-and-understanding) (3 papers)
- [Topic 4: Reasoning and Problem Solving](#topic-4-reasoning-and-problem-solving) (6 papers)
- [Topic 5: Cultural Understanding and Multilingualism](#topic-5-cultural-understanding-and-multilingualism) (4 papers)
- [Topic 6: Safety and Reliability in LLMs](#topic-6-safety-and-reliability-in-llms) (6 papers)
- [Topic 7: Natural Language Processing and Understanding](#topic-7-natural-language-processing-and-understanding) (10 papers)
- [Topic 8: Specialized Domain Applications](#topic-8-specialized-domain-applications) (9 papers)
- [Topic 9: Automated and Enhanced Learning](#topic-9-automated-and-enhanced-learning) (5 papers)
- [Topic 10: Miscellaneous](#topic-10-miscellaneous) (48 papers)

---

## Topic 1: Memory Efficiency and Compression

### 主题概述
内存效率和压缩技术是大型语言模型（LLMs）优化的关键领域之一。随着模型规模的不断增大，它们在处理推理任务或长文档时所需的内存资源也相应增加，这不仅限制了模型的部署灵活性，还增加了计算成本。因此，提高内存效率和开发有效的压缩方法对于推动LLMs在实际应用中的性能提升和规模化部署至关重要。这些技术有助于降低模型的内存占用，同时保证其推理能力和响应质量，尤其适用于边缘设备和需要长期记忆能力的应用场景，如多轮对话和医疗健康服务。

### 各论文贡献
- 来自Westlake University和Zhejiang University的Wenjie Du等人提出了RLKV框架，主要贡献是通过引入“推理头”的概念，并利用强化学习指导KV缓存压缩，实现了在维持高性能的同时减少内存开销。在GSM8K、Math500、AIME24和MBPP等数据集上进行的实验显示，与现有KV缓存压缩方法相比，该方法在不同稀疏度水平下均表现出色，显著降低了内存使用量，而性能下降最小[^0]。
  
- 来自Northeastern University的Xin Liu等人提出了一种名为Semantic-Anchor Compression (SAC)的方法，其独特之处在于通过选择语义上有代表性的“锚点令牌”并增强它们，来避免需要大量预训练的自动编码过程。这种方法还采用了双向注意力机制，提高了压缩效果。实验结果显示，在SlimPajama-6B和MRQA数据集上，SAC相比Full-FT、LLMLingua-2、ICAE、500xCompressor和EPL等基线方法，在ROUGE-1 F1和Exact Match (EM)指标上均有显著提升[^49]。

- 来自Kaliber AI Research Division的Aneesh Jonelagadda等人设计并实现了一个无监督的人类启发式长期记忆架构Mnemosyne，旨在改进边缘设备上LLMs的长期记忆能力。该架构结合了模块化的摄入过滤器、动态图结构存储以及模拟人类记忆召回动态的过程，解决了现有方法在处理重复性和语义相似但时间不同的对话时遇到的挑战。实验表明，在LoCoMo基准测试数据集上，Mnemosyne相比于RAG、Mem0、Zep、MemGPT/LangMem、OpenAI和Memory-R1等基线方法，在J-score和人类评价胜率方面都有明显改善[^72]。

### 技术趋势
该主题下的研究展示了多种技术路线的发展，包括基于强化学习的KV缓存压缩策略、无自动编码需求的语义锚点压缩方法，以及模仿人类记忆管理的长期记忆架构。这些方法共同指向了提高LLMs内存效率和压缩能力的目标，同时努力保持或甚至提升模型性能。可以看出，研究者们正积极探索如何将人类认知特点融入到LLMs的设计中，以克服固定上下文限制带来的问题，尤其是在需要长时间跟踪和记忆的任务中。

### 数据集和评估
- 数据集：GSM8K、Math500、AIME24、MBPP、SlimPajama-6B、MRQA、LoCoMo基准测试数据集。
- 评估指标：涵盖了性能比较、内存效率、错误模式分析、ROUGE-1 F1、Exact Match (EM)、J-score和人类评价胜率等，这些指标从多个维度衡量了模型的记忆效率和压缩后的表现。

---

## Topic 2: Legal and Regulatory Applications

### 主题概述
法律和监管应用是人工智能领域的重要分支，它旨在通过利用机器学习技术和自然语言处理来提高法律文件分析的效率和准确性，尤其是在检测不公平条款、区分人机生成文本等方面发挥重要作用。这些技术的应用不仅有助于提高工作效率，还能促进透明度、保护消费者权益，并确保企业遵守相关法规。此外，它们还能够识别由大型语言模型生成的内容，这对于维护数字通信的信任和防止滥用至关重要。

### 各论文贡献
- 来自Technical University of Munich的Hyunji Lee等人提出了结合蒙特卡洛树搜索（MCTS）与代理提示评估器的框架，用于优化法律文本分类中的提示设计。主要贡献在于通过集成MCTS实现高效的提示空间探索，开发代理提示评估器以减少大型语言模型（LLMs）的推理成本，并利用文本梯度进行提示迭代优化。在CLAUDETTE数据集上的实验表明，相较于SVM、零样本模型、GrIPS和OPRO等基线方法，该框架在检测服务条款中的公平性和非公平性方面表现出了更高的准确率和宏观F1值[^1]。
- 来自Dialpad Inc.的Elena Khasanova等人提出了基于阅读理解的领域适应持续指令预训练方法（DACIP-RC），针对商业对话任务进行了优化。该工作的独特之处在于首次将指令预训练应用于商业对话数据，并采用阅读理解技术生成多样化的任务指令和响应，从而克服了大规模LLMs高推理成本以及小型LLMs在多领域零样本泛化能力不足的问题。在QMSUM、PubMedQA和MediQA-QS等数据集上的实验显示，与传统微调方法和内部NTP模型相比，DACIP-RC在F1-Score、ROUGE-2和BERTScore等指标上均表现出显著的性能提升[^19]。
- 来自MBZUAI的Cong Zeng等人提出了将检测AI生成文本的任务重新定义为分布外检测问题的方法。该方法的核心创新在于将人类撰写的文本视为分布异常值，而将机器生成的文本作为分布内样本，进而通过一元分类方法（如DeepSVDD和HRN）和基于得分的学习技术（如能量方法）构建检测框架。实验结果表明，在DeepFake、M4-multilingual和RAID数据集上，这种方法相较于DetectLLM、DetectGPT等现有基线方法在AUROC、AUPR和FPR95等指标上有明显改善，显示出更强的跨领域和跨模型泛化能力[^70]。

### 技术趋势
在法律和监管应用的研究中，可以看到几个显著的技术趋势：首先，对于特定任务如法律文本分类，研究者们正在探索如何更有效地利用大型语言模型（LLMs）的潜力，包括通过优化提示设计来增强模型的表现；其次，为了应对大规模模型在实际部署中的高计算成本，领域适应和持续学习成为关键方向，旨在使较小规模的模型也能在多种业务场景中展现出良好的零样本泛化能力；最后，随着AI生成内容的增多，如何可靠地识别这些内容成为了另一个重要的研究方向，其中，将人类撰写的文本视为异常值的分布外检测方法展示了其独特的价值和优势。

### 数据集和评估
- 法律文本分类任务使用了CLAUDETTE数据集，评估指标包括准确率和宏观F1值。
- 商业对话任务采用了Dialpad Inc.的专有商业对话转录数据集，以及公开的QMSUM、PubMedQA和MediQA-QS数据集，评价标准涵盖F1-Score、ROUGE-2、BERTScore和用户偏好调查。
- AI生成文本检测任务涉及DeepFake、M4-multilingual和RAID数据集，评价指标包括AUROC、AUPR和FPR95。这些数据集和评估标准共同构成了该领域研究的基础，帮助研究人员量化和比较不同方法的有效性。

---

## Topic 3: Multimodal Reasoning and Understanding

### 主题概述

多模态推理与理解（Multimodal Reasoning and Understanding）是指通过结合文本、图像等多种形式的信息，来实现更加复杂和精准的推理任务。这一领域的研究对于提高人工智能系统处理现实世界问题的能力至关重要，特别是在需要综合多种类型数据以做出决策或解决问题的情境中。例如，在医疗诊断、自动驾驶等领域，多模态推理能够帮助系统更好地理解和利用不同来源的信息，从而提高其性能和可靠性。

### 各论文贡献

- 来自University of California, Los Angeles的Shuang Chen等人提出了ARES：一种通过难度感知的令牌级熵形塑实现多模态适应性推理的方法。该方法的主要贡献在于引入了高窗口熵（HWE）令牌作为探索触发器，并设计了一种层次化的奖励机制，根据任务难度调整探索深度，以及动态分配思考预算的令牌适应性KL设计。这些创新点旨在优化多模态语言推理模型（MLRMs）在涉及文本和视觉数据的复杂推理任务中的表现和推断效率。在大约224K样本的数据集上进行冷启动阶段的训练，并在ViRL39K数据集上进行了RLVR阶段的测试，结果表明ARES相较于基线模型如GPT-4.1等，在Pass@1准确性方面有所提升[^7]。

- 来自School of Information Science and Technology, VISTEC的Tinnakit Udsa等人探讨了大型语言模型在联邦学习环境下的跨客户端训练数据记忆问题。他们的主要贡献在于开发了一种框架，用于测量联邦学习中的跨客户端记忆情况，包括引入了成对记忆比率($\mathcal{\sf MR}_{j\rightarrow k}$)和相关记忆度量指标，以量化同一客户端内和跨客户端的记忆程度。这解决了现有联邦学习记忆检测技术难以捕捉真实记忆泄漏的风险问题，使大型语言模型在如医疗保健等隐私敏感领域应用时更安全可靠。通过对总结、对话、问答和分类四个数据集上的实验，展示了该方法相较于传统的基线方法，如Canary Injection和Verbatim Memorization等，在$\mathcal{\sf MR_{Intra}}$和$\mathcal{\sf MR_{Inter}}$等评估指标上有了显著的进步[^89]。

- 来自Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)的Tajamul Ashraf等人提出了MATRIX：一种针对工具使用推理的多模态代理调优框架。该方法通过结合轨迹驱动的监督微调和偏好优化两个阶段来提升多模态语言模型在复杂推理和决策任务中的有效性。其独特贡献在于提出了一种自动化合成和验证管道（M-TRACE），用于生成高质量的训练轨迹，并引入了Pref-X，通过偏好标注步骤对齐来改善工具使用的决策过程。实验结果显示，MATRIX在多个数据集上，比如M-TRACE、Pref-X等，相较于封闭源代码控制器（如GPT-4）和开源控制器（如Qwen2-VL-7B），在逐步正确性、深层推理连贯性和事实准确性、整体任务解决成功率等方面表现出色[^98]。

### 技术趋势

在多模态推理与理解的研究中，技术趋势逐渐向更复杂的模型结构和更精细的训练策略发展。ARES通过引入基于熵的探索控制机制，实现了根据任务难度动态调整推理努力程度的目标；而MATRIX则通过结合大规模多模态任务数据集的监督微调和偏好优化，提升了模型在实际应用场景中的鲁棒性和适应性。这两项工作都反映了当前研究试图克服传统方法中数据稀缺、手动标注成本高、泛化能力有限等问题的趋势，向着更加高效、精确且具有广泛适用性的多模态推理模型迈进。

### 数据集和评估

在所提及的论文中，研究人员使用了多种多样的数据集，包括专注于特定领域任务的数据集（如STEM任务、ViRL39K）、以及用于评估联邦学习环境下记忆风险的总结、对话、问答和分类数据集。此外，还有专门用于测试多模态代理工具使用能力的数据集M-TRACE。评估指标涵盖了从任务解决的成功率到推理的准确性等多个维度，如Pass@1准确性、平均@16准确性、$\mathcal{\sf MR_{Intra}}$、$\mathcal{\sf MR_{Inter}}$、答案准确性（AnsAcc）等，确保了模型在不同场景下的性能可以被全面地衡量和比较。

---

## Topic 4: Reasoning and Problem Solving

### 主题概述

**推理与问题解决（Reasoning and Problem Solving）** 是人工智能领域的一个关键主题，特别是在自然语言处理（NLP）和大型语言模型（LLMs）的发展中占据核心地位。随着LLMs在各种任务中的应用越来越广泛，如何确保这些模型不仅能够记忆训练数据，还能真正具备理解和解决问题的能力，成为了一个亟待解决的问题。此外，如何通过有效的基准测试来衡量和提高模型的推理能力，也是该领域的重要议题。本主题下，各论文分别从自动基准测试进化、多轮搜索代理的动态上下文管理、LLM生成的理由对人类判断的影响、多智能体系统驱动的自动化形式化以及改进离散扩散语言模型等方面进行了深入探讨。

### 各论文贡献

- 来自University of California, Davis的Qin Liu等人提出了ArenaBencher，一种用于自动基准测试进化的模型无关框架，其主要贡献在于利用多个模型的反馈来生成和验证新的测试案例，从而减轻偏见和过拟合，设计了针对广泛挑战性测试案例的能力感知更新机制，并通过迭代精炼策略增强挑战性和诊断相关性[^3]。在包括GSM8K、AdvBench Harmful Behaviors和CommonsenseQA等数据集上的实验表明，相比原始基准版本和单模型反馈变体，ArenaBencher显著提升了准确性、攻击成功率、公平性、分离性、一致性以及难度。

- 来自Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences的Qiaoyu Tang等人开发了DeepMiner，旨在通过动态上下文窗口管理改善多轮搜索代理的性能，解决了现有数据集中任务复杂度不足以及上下文管理限制多轮有效交互次数的问题[^13]。实验结果表明，使用Qwen3-32B模型，DeepMiner在HotpotQA、TriviaQA、2WikiMultihopQA、BrowseComp、BrowseComp-zh、XBench-DeepSearch和GAIA等多个数据集上实现了更高的准确率，并支持更多的交互轮次，超越了商业系统和先进的一般模型。

- 来自University of Maryland, College Park, USA的Shramay Palta等人研究了大型语言模型生成的理由如何影响人类对常识推理任务答案的合理性判断，揭示了LLM生成的理由对人类判断的潜在影响及其可能带来的偏差[^26]。通过对SIQA和CQA数据集上的实验，展示了LLM生成的理由可以显著改变人类对答案合理性的评价，特别是在PRO和CON理由条件下，这为评估常识推理的可靠性提供了新视角。

- 来自University of Southern California的Gregory Yauney等人提出了MDAD（最小可检测能力差异），作为一种新的元评估指标，以提高语言模型微基准测试的可靠性并反映完整基准测试的表现，解决了微基准测试是否能准确反映模型性能的问题，同时评估了随机采样相对于复杂微基准测试方法的有效性[^56]。实验表明，在MMLU、BIG-Bench Hard (BBH)、MMLU-Pro和GPQA等数据集上，MDAD提供了一种更细致且可靠的评估方式，能够更加精确地衡量模型之间的表现差异。

- 来自Massachusetts Institute of Technology的Cai Zhou等人提出了一种分层扩散语言模型（HDLM），通过引入分层词汇表和时间变化的下一个规模预测过程，解决了现有离散扩散语言模型缺乏丰富语义和在去噪过程中出现不一致的问题，同时也克服了自回归模型无法有效修订先前生成的标记的局限[^60]。在OpenWebText数据集上的实验显示，HDLM相较于GPT2、Llama110M、SEDD、MDLM-small和GIDD+-small等基线模型，在验证困惑度和生成困惑度方面均有显著提升，展现了其在语言生成质量与灵活性方面的优势。

### 技术趋势

该主题下的论文普遍关注于提高模型的推理能力和问题解决能力，通过不同的技术和方法来实现这一目标。其中，ArenaBencher和DeepMiner采用了多模型反馈和动态上下文管理技术来优化和扩展模型的性能；而MASA则通过构建一个多智能体系统来实现数学推理的自动化形式化。此外，Shramay Palta等人的研究探索了LLM生成的理由对人类判断的影响，进一步理解了模型解释与人类认知之间的关系。Gregory Yauney等人的工作引入了新的元评估指标MDAD，以提升微基准测试的可靠性和效率。最后，Cai Zhou等人通过分层扩散模型改进了离散扩散语言模型的语义丰富性和去噪一致性。

### 数据集和评估

论文中使用的主要数据集包括数学推理、常识推理、安全性、深度搜索以及语言生成相关的多个公开数据集，如GSM8K、CommonsenseQA、HotpotQA、TriviaQA、BrowseComp、GAIA、SIQA、CQA、MMLU、BIG-Bench Hard (BBH)、MMLU-Pro、GPQA和OpenWebText等。评估指标方面，除了通用的准确性之外，还包括了攻击成功率（ASR）、公平性、分离性、一致性、难度、验证困惑度（Valid. PPL）、生成困惑度（Gen. PPL）、平均合理性评分、变化评分、卡方同质性检验以及一些特定领域的评估标准，如BLEU-4、ChrF、RUBY、通过率、形式化正确性（FC）等。


---

## Topic 5: Cultural Understanding and Multilingualism

**主题概述**

本主题聚焦于文化理解与多语言能力的研究，探讨如何通过先进的自然语言处理技术和框架来提高机器学习模型在跨文化和多语言环境中的表现。随着全球化进程的加速和技术的进步，能够理解和处理多种语言及文化背景信息的系统变得越来越重要。这些研究不仅有助于提升人工智能系统的文化敏感性和语言多样性，还能促进不同文化之间的交流和理解。

**各论文贡献**

- 来自东北大学的Shuliang Liu等人提出了Group-Based Polling Optimization (Genii)，主要贡献在于开发了一个无监督的多代理协作优化框架，用于减轻大型语言模型（LLMs）作为评估者的判断偏好偏差问题。通过模拟客户端-服务器投票机制，Genii利用组内一致性评分来优化每个代理，从而实现集体偏好的形成。在包括Evol-Instruct、UltraFeedback等在内的多个数据集上进行的实验显示，与传统的基准方法如Vanilla LLM和Self-Consistency相比，Genii显著降低了有害自我偏好倾向(HSPP)，提高了准确性[^22]。

- 来自Yaraku, Inc.的Vincent Michael Sutanto等人研究了ChatGPT作为日语-英语翻译工具的应用潜力，探索了不同提示策略、文档级与句子级翻译方式以及不同版本ChatGPT的表现。这项工作的独特之处在于它提供了ChatGPT与其他专业翻译引擎之间详尽的比较分析，同时开发了一种基于MQM框架的人类评估工具。实验结果显示，通过改进提示策略，ChatGPT的翻译质量得到了显著改善，尤其在ParaNatCom和FLORES等数据集上的表现优于某些商业系统，自动评估指标如BLEU和COMET得分也有所提升[^27]。

- 来自新加坡科技设计大学的Weihua Zheng等人开发了MMA-ASIA框架，旨在评估大型语言模型在亚洲文化背景下多模态理解的能力。该框架的主要创新点包括首次创建了涵盖文本、图像和语音三种模式且在输入层面进行对齐的多语言数据集，以及五维评价协议，确保全面的文化意识评估。实验中使用了包含27,000个问题的MMA-ASIA数据集，涉及十种语言。结果表明，通过引入VPR方法消除推理幻觉，MMA-ASIA在GPT-4o、Qwen等模型上实现了文化知识的更好泛化和跨语言、跨模态的一致性，超越了现有文化的基准数据集[^71]。

- 来自南洋理工大学的Yuxin Li等人提出了HAREN-CTC方法，用于从语音信号中检测抑郁症状。该方法的主要创新在于利用层次化的自监督表示学习来捕捉语音中的异质抑郁线索，并结合连接主义时间分类(CTC)进行弱监督学习，无需帧级别的注释。通过整合多层SSL特征并运用交叉注意力机制，HAREN-CTC成功提升了抑郁检测的准确率和泛化能力。在DAIC-WOZ和MODMA数据集上的实验表明，与现有基线方法如DepAudioNet和Speechformer相比，HAREN-CTC在宏观F1分数、召回率和精确度上均有明显提升[^77]。

**技术趋势**

上述论文展示了自然语言处理领域内针对文化理解和多语言能力提升的不同技术路径。Genii侧重于通过多代理协作优化来减轻模型的判断偏差；ChatGPT的案例研究强调了提示工程和多模态数据处理的重要性；MMA-ASIA则致力于构建全面的文化意识评估框架，以促进跨语言和跨模态的理解；而HAREN-CTC则展示了自监督学习方法在特定应用领域的潜力，特别是抑郁症检测。这些研究共同推动了多语言模型的文化适应性和整体性能的发展，体现了技术向更加细致、复杂方向演进的趋势。

**数据集和评估**

这些论文采用了多样化的数据集和评估指标，反映了其研究目标和应用场景的广泛性。Genii利用了Evol-Instruct、UltraFeedback等多个数据集，评估指标包括准确性和有害自我偏好倾向(HSPP)。ChatGPT的翻译研究使用了ParaNatCom、FLORES等数据集，评估指标涵盖了BLEU、COMET等自动评分标准及MQM人类评分。MMA-ASIA构建了自己的同名数据集，评估文化意识的维度包括准确性、跨语言一致性、跨模态一致性等。最后，HAREN-CTC在DAIC-WOZ和MODMA数据集上测试，主要使用宏观F1分数、召回率和精确度作为评估标准。这些数据集的选择和评估指标的多样化，进一步证实了研究者们对于提升模型在实际应用中可靠性的重视。

---

## Topic 6: Safety and Reliability in LLMs

### 主题概述

大型语言模型（LLMs）的安全性和可靠性是当前人工智能领域的一个重要研究方向。随着LLMs的应用越来越广泛，如何确保这些模型能够提供安全、可靠的信息，并且避免有害或误导性的响应变得至关重要。这一主题不仅涉及技术层面的改进，还包括伦理和社会责任方面的考量，对于推动LLMs的健康发展具有重要意义。

### 各论文贡献

- 来自北京邮电大学的Muxi Diao等人提出了AutoRed框架，主要贡献是实现了无需依赖种子指令的自由形式对抗性提示生成，以及通过人格引导来增加提示的语义多样性。实验显示，在AutoRed-Hard和AutoRed-Medium数据集上，该框架相比其他自动化红队方法如StrongR、Beaver等显著提高了攻击成功率（ASR）[^11]。

- 来自萨克森人工智能与数据分析中心（ScaDS.AI）和德累斯顿工业大学的Shuzhou Yuan等人开发了Exaggerated Safety Benchmark（XSB）和Multi-turn Scenario-based Exaggerated Safety Benchmark（MS-XSB），并提出了一系列轻量级、模型无关的缓解策略，旨在减少大型语言模型过度拒绝良性请求的现象。实验结果表明，这些策略在提高合规率的同时减少了拒绝率，相较于无缓解措施的方法有了明显的改善[^17]。

- 来自乌得勒支大学的Haoyang Gui等人研究了LLMs在社交媒体影响者营销中的监管合规应用，尤其是未披露赞助内容的检测。他们提出了一个结合法律推理能力的分类框架，通过不同层次的法律知识输入，评估LLMs生成解释的质量。实验结果显示，该框架在检测准确性上优于传统的逻辑回归方法，并提供了详细的错误类型分析[^23]。

- 来自淘宝天猫集团和清华大学的Jianhui Yang等人设计了适应性引导强化学习框架（TaoSR-AGRL），用于优化电商搜索相关性系统。此框架解决了传统强化学习方法中的奖励稀疏和探索停滞问题，通过规则感知的奖励塑造和适应性引导重放机制，提高了模型在复杂查询处理中的准确性和可解释性。实验表明，TaoSR-AGRL在多个评价指标上均优于基线方法，包括TbStar-DPO、GRPO和GRPO-PR[^44]。

- 来自哈尔滨工业大学的Xianzhen Luo等人进行了首次大规模的经验性研究，分析了代码LLMs的扩展规律。研究发现，与自然语言LLMs相比，代码LLMs需要更高的数据到参数比例才能达到最优性能，且Farseer扩展规律更适合于代码LLMs的性能预测。这项工作对于优化代码LLMs的训练过程和资源分配有着重要的指导意义[^59]。

- 来自纽约大学阿布扎比分校的Nouar Aldahoul等人提出了一种多语言多代理系统，利用检索增强生成（RAG）方法，旨在跨多种语言和各种对抗性攻击情况下有效检测和缓解虚假信息。该系统整合了网络爬虫、管理、虚假信息检测、话题分类和判断代理，展示了其在多个数据集上的鲁棒性，特别是在对抗性攻击场景下的表现超越了基线模型如ChatGPT 4.0 beta、Gemini等[^69]。

### 技术趋势

上述论文展现了几个关键的技术趋势：一是对抗性测试和安全评估，通过生成多样化的对抗性提示来测试和改进LLMs的安全性；二是多任务学习和模型灵活性的提升，尤其是在应对复杂查询和多语言环境下的表现；三是引入法律推理能力和专业知识，以增强LLMs在特定领域的应用能力和合规性；四是优化模型训练过程，特别是针对代码数据的特性，研究其特有的扩展规律，以实现更高效的训练和更好的性能。

### 数据集和评估

论文中使用的数据集包括针对不同应用场景定制的数据集，如用于对抗性测试的AutoRed-Hard和AutoRed-Medium，用于夸张安全行为评估的XSB和MS-XSB，用于法律推理评估的Instagram帖子数据集，以及电商搜索相关性优化所需的平衡评估集和真实世界评估集。此外，还有专门用于代码LLMs扩展规律分析的Opencoder代码预训练数据和内部代码库。评估指标涵盖了攻击成功率（ASR）、合规率、拒绝率、精确度、召回率、F1分数、帮助度评分、错误率、验证损失等，全面反映了LLMs在不同场景下的性能和可靠性。


---

## Topic 7: Natural Language Processing and Understanding

### 主题概述
自然语言处理与理解（Natural Language Processing and Understanding, NLP&U）是人工智能领域的一个关键分支，专注于使计算机能够理解和生成人类语言。随着大语言模型（Large Language Models, LLMs）的发展，NLP&U的研究已经深入到更加复杂的任务，如情感分析、对话生成、概念学习等。这些研究不仅推动了技术的进步，还促进了LLMs在实际应用中的可靠性、可控性和效率的提升。因此，NLP&U对于开发更智能、更适应多变环境的AI系统至关重要。

### 各论文贡献
- 来自Google DeepMind的John Hewitt等人提出了Neologism Learning，主要贡献是在引入新概念给语言模型的过程中，通过词汇扩展和特定训练目标优化，增强了模型的可控性和自我表述能力。在LIMA和AxBench上的实验表明，相比传统方法，新概念的学习和表述有了显著改善[^2]。
- 来自IBM Research爱尔兰分部的Jasmina Gajcin等人提出了一种通过可验证全局解释来解读LLM作为评判者的决策策略的方法，即CLoVE和GloVE。该方法通过对比式局部解释和合成全局政策，解决了LLM评判者模型的透明度问题。实验结果表明，这种方法能有效提高决策过程的可解释性，特别是在多个测试数据集上表现出色[^24]。
- 来自北京航空航天大学的Hainan Zhang等人提出了FedDTRE，一种基于信任度评价的联邦对话生成模型。通过动态调整融合系数和自适应聚合策略，FedDTRE在保护隐私的同时提高了对话系统的性能和个性化水平。在Synthetic-Persona-Chat等多个数据集上的实验显示，其性能优于传统的联邦学习方法FedAvg和FedProx[^25]。
- 来自中央南大学及上海人工智能实验室的Cheng Yang等人设计了一个名为MUSE的记忆利用和自我演化代理框架，用于执行长期生产力任务。MUSE通过计划-执行-反思-记忆循环不断学习和改进，解决了现有AI代理无法持续学习的问题。实验结果显示，在TheAgentCompany (TAC)数据集上，MUSE的完美完成率（PCR）和部分完成得分（$S_{partial}$）均高于基准模型[^28]。
- 来自ALTA Institute的Bianca-Mihaela Ganescu等人开发了一种名为SOP-Maze的复杂业务标准操作程序（SOPs）评估基准，用于评价LLMs在处理复杂业务流程时的能力。该研究通过将任务分类为Lateral Root System (LRS)和Heart Root System (HRS)，揭示了LLMs在理解和执行复杂SOPs方面的局限性。实验表明，SOP-Maze能够有效评估LLMs在实际业务场景中的表现[^48]。
- 来自Inria, INSA Lyon, CITI的Abhishek K. Mishra等人进行了跨语言的大规模语言模型隐私泄露比较分析。他们发现语言结构显著影响隐私泄露，并提出了一系列攻击向量来量化这一影响。实验结果证实了语言结构对隐私风险的重要性，并建议在不同语言环境中采取针对性的隐私保护措施[^54]。
- 来自未指定机构的Wen-Yu Chang等人提出了一个基于轻量级用户行为洞察的销售导向对话代理策略，旨在通过模拟不同用户群体的行为并将其整合到代理的对话策略中，提高销售成功率。实验显示，该策略能够根据用户的年龄、性别和职业等因素，调整对话策略以更好地满足用户需求，从而提升了对话的成功率和效率[^64]。

### 技术趋势
这些论文展示了自然语言处理与理解领域的几个主要技术趋势：一是增强语言模型的可控性和透明度；二是提高对话生成模型的隐私保护和个性化能力；三是通过经验驱动的方式，使AI代理能够持续学习和适应；四是针对特定任务（如业务SOP）和领域（如医疗文本）的定制化模型评估；五是探索用户行为对销售导向对话系统的影响，以实现更有效的对话策略。

### 数据集和评估
论文中使用了多种数据集，包括LIMA、AxBench、Synthetic-Persona-Chat、CMU_DoG、TheAgentCompany (TAC)、HiTZ Multilingual Medical Corpus、以及Persona数据集等，这些数据集涵盖了从语言理解和生成到对话策略和用户体验的不同方面。评估指标也多样化，包括了LLM评分、BLEU、ROUGE、BERTScore、部分完成得分（$S_{partial}$）、完美完成率（PCR）、以及各种隐私泄漏相关指标等，反映了研究者们对模型性能、成本效益和隐私安全的综合考量。

---

## Topic 8: Specialized Domain Applications

### 主题概述
本报告涵盖了几个在特定领域应用中的前沿研究，这些领域包括多语言情感分析、视觉与语言模型融合、大语言模型偏好优化、自动发现可解释特征用于评论质量评估、针对特定语言的大语言模型设计以及新的评估指标和偏见缓解策略。这些研究不仅探索了现有技术和模型的局限性，还提出了一系列创新的方法和技术，以提升模型在特定任务中的表现，特别是在面对多语言、跨文化以及需要创造性文本生成等挑战时。通过引入新数据集、改进模型架构及开发新的评估指标，这些研究为相关领域的未来发展提供了重要的参考价值。

### 各论文贡献
- 来自King Abdullah University of Science and Technology的Qiang Yang等人提出了SenWave，一个细粒度多语言情感分析的数据集，该数据集来源于与COVID-19相关的推特内容。主要贡献在于引入了一个全面注释的数据集，包含多种语言的推特，以支持细粒度的情感分析，解决了现有数据集中缺乏全面标注和细致情感标签的问题[^16]。
  
- 来自University of British Columbia的Jiayun Luo等人提出了DIYSink框架，旨在通过理解和优化视觉注意力机制中的“注意力沉没”现象来提高大型视觉语言模型（LVLMs）的性能。其创新之处在于识别并分析ViT注意力沉没的作用，并开发了一种系统性的方法来根据任务需求选择性地强调或淡化这些沉没标记[^32]。
  
- 来自Stony Brook University的Jason Bohne等人提出了Mix-和MoE-DPO框架，扩展了直接偏好优化（DPO）方法，通过软混合模型和专家混合架构实现对大语言模型（LLMs）更有效的偏好优化。这一方法能够更好地处理多任务和异构偏好设置，解决了标准DPO在多任务场景中表达力不足的问题[^38]。
  
- 来自Tsinghua University的Xiaochong Lan等人开发了AutoQual，一个基于LLM的自动化发现可解释特征框架，用于在线评论的质量评估。其主要创新点在于整合反思、工具实施和记忆架构来发现最有效的特征，解决了手动特征工程不可扩展和深度学习模型缺乏透明度的问题[^38]。
  
- 来自University of California, San Diego的V. S. Raghu Parupudi提出了一个新颖的评估指标——置信分数（CS），用于衡量大语言模型在创意文本生成任务中的表现。这个指标通过考虑输出概率分布的形状来提供更平衡和准确的评价，解决了传统评估指标偏向于稳定但缺乏创意的响应的问题[^74]。
  
- 来自Beijing Institute of Technology的Chuyi Tan等人提出了Reinforcement Learning with Ensembled Rewards (RLER)，一种缓解自我奖励强化学习中系统偏见的技术。该技术通过模型集成构建统一奖励空间，采用自适应软奖励插值和信心-分歧平衡回滚选择策略，解决了由于标注样本有限而导致的奖励估计偏见问题[^80]。

### 技术趋势
这些论文展示了在特定领域应用中的几个关键技术创新趋势：
1. **数据集的定制化**：针对特定场景或语言，构建定制化的数据集以增强模型的针对性和准确性。
2. **模型架构的创新**：引入如专家混合（MoE）、动态令牌选择模块等新型架构，以改善模型在特定任务上的性能。
3. **评估方法的发展**：开发新的评估指标，例如置信分数（CS），以适应更加复杂和多样化的任务需求，尤其是涉及创造性文本生成的任务。
4. **强化学习与偏见缓解**：利用强化学习技术，特别是通过集成多个模型来减轻偏见，提高模型的泛化能力和稳定性。

### 数据集和评估
- **数据集**：SenWave使用了105百万条未标注的推特数据；DIYSink实验覆盖了LLaVA eval、GQA等多个数据集；Mix-和MoE-DPO则使用了IMDb和Amazon Book Reviews数据集；AutoQual采用了Amazon review和Meituan私有数据集；Quality Estimation Reranking专注于WMT23测试集；而Diagnosing and Mitigating System Bias in Self-Rewarding RL则使用了Arithmetic Dataset、DAPO-Math-17K等数据集。
  
- **评估指标**：SenWave使用了诸如准确率、F1-Macro等传统分类指标；DIYSink采用了每个数据集特定的性能指标；Mix-和MoE-DPO则关注于情感、信息量和语法的评价；AutoQual采用了Spearman’s Rho、Mean Absolute Error等；Quality Estimation Reranking使用了BLEURT-20、COMET-22等；Diagnosing and Mitigating System Bias in Self-Rewarding RL则通过Avg@8、Test Accuracy等指标来评价模型性能。

---

## Topic 9: Automated and Enhanced Learning

**主题概述**

自动化与增强学习（Automated and Enhanced Learning）是当前人工智能研究中的一个重要领域，它涉及利用预训练模型进行参数高效微调（PEFT）、通过强化学习改进大型语言模型的推理能力、以及利用自然语言处理技术自动解析和生成软件项目中的治理文档和用户故事。此外，该领域还探索了AI在学术出版物同行评审过程中的应用。这些研究对于提高机器学习模型的适应性和效率、优化软件开发流程、以及增强学术出版的质量和一致性具有重要意义。

**各论文贡献**

- 来自Meta的Md Kowsher等人提出了SliceFine，主要贡献是引入了“通用胜利切片假设”(Universal Winning Slice Hypothesis, UWSH)，并通过仅更新选定权重切片的方法来实现参数高效的微调，从而解决了确定预训练模型适应性背后机制的问题，并展示了新参数是否必要。在包括VTAB-1K等在内的多个数据集上的实验表明，与诸如LoRA、AdaLoRA等基线方法相比，SliceFine提高了训练速度，降低了内存使用，并使模型更加紧凑[^30]。

- 来自Bitdefender的Marius Dragoi等人提出了一种新的度量标准Cover@$\tau$，用于评估基于强化学习的验证奖励(RLVR)模型在不同可靠性水平下的推理边界，解决了传统Pass@k度量标准可能误导评估推理能力的问题。实验结果表明，Cover@$\tau$能更全面地评价不同RLVR方法的推理能力，相较于传统的Pass@1度量标准，提供了更加细致的性能评估视角[^36]。

- 来自University of California Davis的Mobina Noori等人开发了一个大规模的时间序列基准，用于研究开源软件项目在广泛采用AI工具之前的集体治理结构。这一工作的主要创新在于使用自然语言处理技术从政策文件中自动提取制度声明，以建立可复制和扩展的治理演变研究框架，填补了缺乏AI影响前治理结构实证研究的空白。通过分析涵盖2013年至2022年的710个仓库，研究人员能够量化治理变化的程度，这为未来的研究提供了宝贵的参照点[^46]。

- 来自EURECOM的Francesco Dente等人提出了一种名为Text2Stories的任务和度量标准，用于量化从利益相关者访谈记录自动生成的用户故事与原始访谈之间的对齐程度。这项工作解决了评估生成的故事与原始输入忠实度的难题，创新性地引入了正确性和完整性作为衡量标准，同时提出了一种基于嵌入的阻塞机制来减少计算成本。在包含17个软件项目的手动注释评估故事的数据集上进行的实验显示，Text2Stories能够有效地检测故事是否准确反映了访谈内容，这超越了现有质量框架仅关注故事书写质量的局限性[^65]。

- 来自Mila – Quebec AI Institute的Gaurav Sahu等人提出了ReviewerToo框架，旨在系统化研究和部署AI辅助的同行评审。该框架将同行评审视为一个社会技术过程，模拟了从手稿摄入到最终合成的整个评审流程。主要创新之处在于通过模块化设计提供了一个透明且可控的AI辅助评审方法，并在大规模数据集上进行了验证，提供了关于AI在评审过程中优势和限制的经验洞察。实验表明，ReviewerToo框架在分类任务上的表现优于传统机器学习模型如XGBoost和BERT Fine-Tuned Classifier，同时也提供了与人类评审员决策的一致性评估[^84]。

**技术趋势**

上述论文展示了自动化与增强学习领域的几个关键趋势：一是参数高效的微调方法的发展，旨在通过修改现有网络结构中的特定部分来提升下游任务性能；二是针对特定应用场景，如逻辑推理和用户故事生成，提出新的度量标准或评估框架；三是将自然语言处理技术应用于复杂文本分析和自动化任务中，例如解析软件治理文件和评估用户故事的准确性；四是将AI技术引入学术同行评审过程，以提高评审的一致性和效率，同时确保透明度和可信度。

**数据集和评估**

- 数据集方面，这些研究涵盖了多个领域内的广泛数据集，包括视觉任务数据集(VTAB-1K等)、逻辑推理问题集(OMEGA, Reasoning Gym)、开源软件项目治理文档集(710个仓库)、软件项目访谈记录集(17个项目及更多)、以及学术同行评审数据集(ICLR-2k Dataset)。
  
- 评估指标则包括了精度、吞吐量、内存使用、训练时间、Cover@$\tau$、$\mathrm{AvgAUC}^{+}_{\text{cov}}$、归一化香农熵、Jensen-Shannon散度、丰富度、宏观F1分数、正确性、完整性、五类分类性能、接受/拒绝二元任务、精确率、召回率、F1分数、准确率、假阳性率、Cohen’s κ、ELO评级等多种形式的指标，以全面评估所提方法的有效性、效率以及其在实际应用中的表现。

---

## Topic 10: Miscellaneous

### 主题概述

这一系列的研究集中在大型语言模型（LLMs）及其相关应用领域中的多个问题上，包括文化理解、法律推理、生物医学实体识别、测试场景中的行为分析等。这些研究不仅揭示了现有模型的局限性和潜在偏见，还提出了一系列创新的方法和技术来改进和扩展LLMs的能力，以更好地适应现实世界的应用需求。通过这些工作，研究人员旨在提高LLMs的可靠性、公平性和安全性，使其在各个领域的实际部署中发挥更大的作用。

### 各论文贡献

- 来自Fondazione Bruno Kessler的Elisa Leonardelli等人提出了LeWiDi-2025，一个学习与分歧共享任务的第三版，主要贡献在于扩展了包含丰富人类判断差异性的基准，并引入了新的评估指标，如曼哈顿距离、沃瑟斯坦距离和平均标准化绝对距离。在四个不同的数据集（Conversational Sarcasm Corpus, MultiPICo, Paraphrase Detection, VariErr NLI）上的实验表明，相比传统的随机基线和最频繁标签基线，新方法能够更有效地捕捉人类判断的多样性[^6]。

- 来自纽约大学的Ioana Marinescu等人开发了一种优化算法来系统地探索标签表示和学习之间的交互作用，主要贡献在于揭示了标签表示和学习在上下文学习中的独立作用，并提供了寻找最优标签集的系统化方法。在情感分类数据集上的实验表明，通过调整标签集的语义相关性，可以显著改善大语言模型的上下文学习性能，特别是在多示例学习和零样本学习中[^8]。

- 来自MaiNLP中心的Jasmin Orth等人提出了一项关于大型语言模型条件接受性判断的研究，主要贡献在于详细探讨了不同LLMs如何整合条件概率和语义相关性，并分析了提示策略的影响。实验结果显示，LLMs在处理条件语句时表现出与人类相似但也有差异的行为模式，特别是在处理复杂逻辑推理方面，这有助于改进其推理能力[^9]。

- 来自匹兹堡大学的Yukai Song等人提出了两阶段投票架构，用于增强社交媒体自杀风险检测的效率和准确性。主要贡献在于结合轻量级模型（如BERT）和大型语言模型（如GPT-5）的优点，减少冗余的LLM调用并提高检测的准确性。实验结果表明，这种方法在Reddit和DeepSuiMind数据集上的表现优于单独使用BERT或GPT-5的方法[^10]。

- 来自东京大学和RIKEN的Taisei Yamamoto等人介绍了CULNIG，一种基于梯度评分的文化神经元识别管道，主要贡献在于能够精确识别出文化和通用文化相关的神经元，排除仅响应任务相关线索或表面文化标记的神经元。实验结果显示，在BLEnD和CulturalBench等数据集上，该方法能够有效提高模型的文化理解能力，比随机神经元屏蔽和原始模型性能基线表现更好[^12]。

- 来自海森人工智能学院的Tim Hagen等人开发了Concausal News Corpus（CCNC），并扩展了因果关系提取任务以包括反因果声明。主要贡献在于解决了当前因果关系提取模型忽视或误分类反因果声明的问题，通过创建包含反因果声明的数据集来训练和评估模型。实验表明，在CCNC和Causal News Corpus v2数据集上，该方法提高了模型区分因果、反因果和非因果陈述的能力[^14]。

- 来自Meta公司的Jingyu Zhang等人提出了一种名为WaltzRL的多智能体强化学习框架，用于协同优化安全性和帮助性。主要贡献在于通过动态改进奖励机制促进两个代理之间的协作，从而减少不安全回应和对良性提示的过度拒绝。实验结果表明，与单一防御模型和无训练时间协作基线相比，该方法显著提高了攻击成功率和过拒绝率指标[^15]。

- 来自俄亥俄州立大学的Jian Xie等人提出了ARM2，一种支持视觉理解和可执行代码的自适应推理模型。主要贡献在于通过引入五种不同的推理格式，利用监督微调（SFT）和强化学习（RL）来适应性选择推理方式，同时支持多模态推理。实验显示，ARM2在多个数据集上表现出色，特别是数学推理和网页搜索任务，比直接提示和简单的推理技术有了明显改进[^18]。

- 来自上海人工智能实验室的Shangheng Du等人开发了AutoMLGen，一种用于机器学习工程任务的自动机器学习生成器。主要贡献在于整合了一个精心策划的机器学习知识库，并采用了蒙特卡洛图搜索（MCGS）来生成和优化机器学习管道。实验结果表明，AutoMLGen在MLE-Bench数据集上相较于其他基线方法，如MLAB和GPT-4o，具有更高的操作灵活性和可重复使用性[^31]。

- 来自北京大学的Haolin Yang等人介绍了NavSpace，一个用于导航代理的空间智能指令评估基准，并开发了SNav模型。主要贡献在于提供了一套完整的评价和提升导航代理空间智能能力的方法。实验结果证明，SNav模型在NavSpace数据集上相比于其他导航模型和多模态大型语言模型表现更为出色，成功提升了导航误差和成功率指标[^42]。

- 来自哥伦比亚大学的Nicholas Deas等人开发了一种名为RefDiv的测试协议，主要贡献在于揭示了大语言模型在测试时由于多样性降低而产生的新型失败模式，并展示了这种方法生成的对抗性提示可以在不同的测试策略和封闭源模型之间成功转移。实验结果表明，RefDiv协议在AdvBench数据集上比其他基线方法如GCG和AutoDAN显著提高了攻击成功率，揭示了测试时间缩放方法的新漏洞[^79]。

- 来自普林斯顿大学的Wouter Haverals等人进行了一个控制实验，比较了人类和AI评估者在文学风格判断中归因偏见的存在情况。主要贡献在于首次系统性地对比了人类和AI评估者的归因偏见，发现AI评估者不仅复制了人类的偏见，而且放大了这种偏见。实验结果表明，在Queneau的作品及其AI变体数据集上，AI评估者对于人类作品的偏好高于AI生成的作品，这一发现具有广泛的学术价值[^86]。

- 来自哥伦比亚大学的Nikhil Reddy Varimalla等人开发了VideoNorms，一个视频语言模型跨文化规范理解的基准测试。主要贡献在于首次针对视频内容的跨文化规范理解进行了系统性的评估，使用了人类-人工智能合作框架进行数据集构建。实验结果显示，在VideoNorms数据集上，视频语言模型能够更准确地预测文化规范的遵守或违反情况，比基线方法有了显著提高[^97]。

### 技术趋势

这些研究展现了多种技术趋势，包括：

- **多模态处理**：许多研究集中在将视觉和其他形式的信息融入到文本处理中，例如，NavSpace和SpatialLadder通过视觉输入增强了导航和空间推理能力。
  
- **多阶段训练**：一些研究采用了多阶段的训练方法来逐步提升模型的特定能力，如ARM2使用了监督微调和强化学习的组合。
  
- **轻量化优化**：部分研究专注于开发无需大量重新训练数据的优化方法，比如Recover-LoRA和BaldWhisper，它们通过轻量级的适配器层或结构简化来恢复或提升模型性能。
  
- **系统化诊断与评估**：研究中广泛运用了系统化的评估和诊断工具，如Systematic Diagnosis of Brittle Reasoning和The Alignment Waltz，分别用于诊断推理中的脆弱环节和评估模型的安全性与帮助性。

### 数据集和评估

论文中使用的数据集和评估指标涵盖了广泛的领域，包括但不限于：

- **数据集**：
  - Conversational Sarcasm Corpus, MultiPICo, Paraphrase Detection, VariErr NLI
  - Sentiment classification datasets
  - Sign Language of the Netherlands (NGT)
  - AdvBench, WikiText-103
  - SpatialLadder-26$k$, ScanNet, SR-91k, VSI-Bench, SPBench-SI, SPBench-MV, CV-Bench, MMSU
  - YpathR, YpathQA-M
  - GutBrainIE challenge dataset
  - LibriSpeech, MuSE, StressID

- **评估指标**：
  - 准确率 (Accuracy)
  - 平均曼哈顿距离 (MAMD)
  - 每次试验累积回报 (Cumulative trial return)
  - 攻击成功率 (Attack Success Rate, ASR)
  - 情感熵 (Semantic Entropy)
  - 语义相似度 (Semantic Similarity)
  - 关键词覆盖率 (Coverage)
  - 链接正确率 (Link Accuracy)
  - 文本错误率 (WER)
  - 精确率 (Precision)
  - 召回率 (Recall)
  - F1分数 (F1 Score)

这些数据集和评估指标的多样性反映了研究团队试图全面覆盖LLMs在不同领域和任务中的应用，从情感识别到因果关系提取，再到专业知识领域中的命名实体识别等。

---

## 参考文献

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


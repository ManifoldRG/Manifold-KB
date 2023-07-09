# Neko Knowledge Base
This repository serves as a knowledge base with key insights, details from other research and implementations to serve as references and one place to document various possible paths to achieve something.

## Papers and their main takeaways

### Multimodal Learning with Transformers Survey ([ArXiv](https://arxiv.org/abs/2206.06488))

- Treat self-attention as graph style modeling, models input sequence as a fully-connected graph. Every modality can be thought of as a graph
	- Text –> every token is a node, and the sequence is the edges connecting them
	- RGB image –> grid graph in pixel space
	- Video & Audio –> clip based graphs over space of temporal and/or semantic patterns
	- Human poses –> key points are nodes, and connections are edges
- Cross-modal interactions are essentially processed by self-attention and its variants
- While implementing transformers, implement pre-normalisation not post (as was done in Vanilla transformer).
- Treat the Transformer-based multimodal pretraining pipelines having three key components, from  bottom to top, i.e., Tokenization, Transformer representation (self-attention variants),  objective supervision (pretraining strategy).
- **Token embedding fusion** – each token could be represented by multiple embeddings. In simplest cases of text, this already happens as for each token we do summation of *token embedding* *⊕* *position embedding*. But there can be more,
	- In VisualBERT, segment embeddings are token-wise added to indicate which modality (vision or language) each token is from
	- In VL-BERT, each token has *linguistic token embedding* *⊕* *full image visual feature embedding*
	- In ImageBERT, five embeddings get fused together – *image  embedding ⊕ position embedding ⊕ linguistic embedding  ⊕ segment embedding ⊕ sequence position embedding*
- **Trends emerging**
	1. Vision language pretraining : image+language, or video+language. Examples: Two-stage (need object detector such as Faster-RCNN) (e.g., LXMERT [103], ViLBert [102], VL-Bert [105]) and end-to-end (e.g., Pixel-Bert  [113], SOHO [203], KD-VLP [204], Simvlm [199]). Two-stage  pipelines have a main advantage – object-aware perceiving.
	2. Speech can be used as text. Many papers end up converting speech to text using ASR and applying the language techniques itself
	3. Currently, most multimodal pretraining works are overly dependent on existence of *well-aligned* multimodal samples, or pairs/tuples. Using weakly-aligned or even unpaired/unaligned multimodal data is still understudied.
	4. Most of the pretraining tasks transfer well across modalities. For e.g. MLM in text domain has been applied to both audio and image, e.g., Masked Acoustic Modeling, Masked Image Region Prediction.
- **Cross-modal Alignment is still a challenge**
	- A representative practice is to map two modalities into  a common representation space with contrastive learning  over paired samples. The models based on this idea are often enormous in size and expensive to optimize from millions  or billions of training data
- **Improving Transferability is a challenge**
	- Transferability : How to  transfer models across different datasets and applications.
	- Data augmentation and adversarial perturbation strategies help multimodal Transformers to improve the generalization ability. VILLA [210] is a two-stage strategy (task-agnostic adversarial pretraining, followed by task-specific adversarial finetuning) that improves VLP Transformers.
	- In real applications, multimodal pretrained Transformers sometimes  need to handle the uni-modal data at inference stage due  to the issue of missing modalities. One solution is using  knowledge distillation, e.g., distilling from multimodal to  uni-modal attention in Transformers [275], distilling from  multiple uni-modal Transformer teachers to a shared Trans-  former encoder [276]


## Dealing with different modalities

### Tokenisation and embedding function

![](media/mml-transformers-modalities.png)

### Self-attention variants

- X<sub>A</sub> and X<sub>B</sub> from two arbitrary modalities,
- Z<sub>(A)</sub> and Z<sub>(B)</sub> denote their respective token embeddings
- `Z` denoting the token embedding (sequence) produced by  the multimodal interactions
- `Tf(·)` stands for the processing  of Transformer layers/blocks.
	- e.g. in “Early Summation”, `Z ← Tf(αZ(A) ⊕ βZ(B)) = MHSA(Q(AB), K(AB), V(AB))`
- Note : these self-attention variants are modality-generic, and can be flexibly combined and nested.
	- TriBERT [183] is a tri-modal cross-attention (co-  attention) for vision, pose, and audio, where given a Query  embedding, its Key and Value embeddings are the con-  catenation from the other modalities


![](media/mml-transformers-self-attention-1.png)
![](media/mml-transformers-self-attention-2.png)

| Self-Attention Variant       | Pros                                                                                                    | Cons                                                                       | Remarks                                                                                                                                                                 | Reference Papers |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| a. Early Summation              | No increase in computational complexity                                                                 | Manually set weights                                                       | Summation of position embedding is a case of early summation                                                                                                            |                  |
| b. Early Concatenation          | Each modality can be encoded well by conditioning the context of other modalities                       | Time complexity                                                            | Also known as All-attenion, or Co-Transformer                                                                                                                           |                  |
| c. Hierarchical Attention (1→2) | TBA                                                                                                     | TBA                                                                        | Multimodal inputs are encoded by independent  Transformer streams and their outputs are concatenated and  fused by another Transformer. Also known as late interaction. |                  |
| d. Hierarchical Attention (2→1) | TBA                                                                                                     | TBA                                                                        | TBA                                                                                                                                                                     |                  |
| e. Cross-Attention              | - attends to each modality conditioned only on the other <br> - no increase in computational complexity | Fails to perform cross-modal attention globally and can lose whole context <br> no self-attention to attend to self-context inside each modality | Query stream is exchanged in a cross-stream manner                                                                                                                      |                  |
|  f. Cross-Attention to Concatenation                           | Alleviates some drawbacks of cross-attention                                                                                                         | Time complexity increases                                                                           | Two streams of cross-attention get concatenated and processed by another transformer to model global context better                                                                                                                                                                         |                  |
- For  enhancing and interpreting the fusion of MML, probing the  interaction and measuring the fusion between modalities  [249] would be an interesting direction to explore

### Pre-training tasks

![](media/mml-transformers-table-3.png)

*Task-Agnostic Multimodal Pretraining*
* Discriminative task oriented pretraining  models do not involve the decoders of Transformer. How to design more unified pipelines that can work  for both discriminative and generative down-stream tasks  is also an open problem to be solved.
* Some practices demonstrate that multi-task training (by adding auxiliary loss) [111], [137] and adversarial training [210] improve multimodal pretraining Transformers to further boost  the performance. However, whether more complexity  is better remains a question.

*Need for Task-Specific Multimodal Pretraining*
- Guhur et al. [150] propose in-domain pretraining for vision-  and-language navigation, as the general VLP focuses on learning vision-language correlations, not designed for se-  quential decision making as required in embodied VLN.
- Special modalities have their own unique domain  knowledge that can be used to design the specific pretrain  pretexts. GraphCodeBERT [44] uses two structure-aware  pretext tasks (i.e., predict where a variable is identified from,  data flow edge prediction between variables) for program-  ming source code.


## Multimodal Datasets

### Usual
* Conceptual  Captions [123]
* COCO [124]
* VQA [125]
* Visual Genome  [126]
* SBU Captions [127]
* Cooking312K [7]
* LAIT [115]
* e-SNLI-VE [128]
* ARCH [129]
* Adversarial VQA [130]
* OTT-  QA [18]
* MULTIMODALQA (MMQA) [131]
* VALUE [132]
* Fashion IQ [133]
* LRS2-BBC [134]
* ActivityNet [135]
* VisDial  [136]

### Million Scale
- Product1M [137]
- Conceptual  12M [138]
- RUC-CAS-WenLan [139] (30M)
- HowToVQA69M  [140]
- HowTo100M [141]
- ALT200M [142]
- LAION-400M  [143]

### More than 2 modalities

- Pano-AVQA [144]
- YouTube-360 (YT-360) [145] (360◦ videos)
- AIST++  [146] (a new multimodal dataset of 3D dance motion and  music)
- Artemis [147] (affective language for visual arts)
- MultiBench [148] provides a dataset including 10  modalities

### Non-QA or captioning
- M3A [151] (financial dataset)
- X-World [152] (autonomous drive)
- MultiMET [153] (a multimodal dataset for metaphor under-  standing)
- Hateful Memes [154] (hate speech in multimodal  memes)
- cooking video YouCookII [155]

## Transformer Applications

### Discriminative
e.g., RGB & optical flow  [46], RGB & depth [213], RGB & point cloud [214], RGB  & LiDAR [215], [216], textual description & point cloud  [31], acoustic & text [180], audio & visual observation for  Audio-Visual Navigation [76], speech query & schema of  SQL database [25], text question/query & the schema SQL  database [24], audio & tags [217], multimodal representation  for video [218], [219], text query & video [220], audio &  video for audio visual speech enhancement (AVSE) [179],  audio & video for Audio-Visual Video Parsing [173], audio  & video for audio-visual speech recognition [134], video &  text for Referring Video Object Segmentation (RVOS) [221], source code & comment & data flow [44], image & text for  retrieval [222].

### Generative
- single-modality to  single-modality
	- e.g., raw audio to 3D mesh sequence [39],  RGB to 3D scene [40], single image to 3D human texture  estimation [223], RGB to scene graph [19], [224], [225], [226],  graph to graph [33], knowledge graph to text [227], video  to scene graph [228], video to caption [229], [230], [231],  [232], image to caption [233], [234], [235], [236], [237], text  to speech [238], text to image [205], [239], text to shape  [240], RGB to 3D human pose and mesh [41], music to  dance [241]
- multimodality to single modality
	- e.g., image  & text to scene graph [242], Video Dialogue (text & audio &  visual to text) [243], Mono Audio & Depth to Binaural Audio  [14], music piece & seed 3D motion to long-range future 3D  motions [146], X-raying image & question to answer [244],  video & text & audio to text [245]
- multimodality to  multimodality (e.g., [246]).

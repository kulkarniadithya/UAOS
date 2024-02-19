<html>
<h1>Weakly Supervised Open-Domain Aspect based Sentiment Analysis (UAOS)</h1>

<h2> Dataset </h2>

The experiments are conducted on SemEval 14, 15, 16 Restaurant and SemEval 14 Laptop datasets.
The original dataset can be found in the data/original_data folder.

<h3> Pre-processing </h3>
Data pre-processing includes four steps:
<ul>
<li>Format Data: The original file is processed and stored in a dictionary.
Run pre-processing/format_data.py to perform this step.</li>
<li>Weak Label Generator: In this step, the CoreNLP dependency parser is executed
to generate the weak labels. 
Download the CORENLP jar files from https://stanfordnlp.github.io/CoreNLP/download.html.
Place stanford-corenlp-4.0.0.jar and stanford-corenlp-4.0.0-models.jar in dependency_parser folder and run the following command from the folder.

```java -Xmx8g -XX:-UseGCOverheadLimit -XX:MaxPermSize=1024m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9015  -port 9015 -timeout 1500000```

Once the process is running on port 9015, run pre-processing/pseudo_labels.py to generate weak labels.
</li>
</ul>

<ul>
<li>Split Data: Once the weak labels are generated for the original train set. This step
splits the train set into pseudo train (reviews for which the weak label generator has identified aspect terms and opinion terms)
and pseudo test (otherwise). The pseudo test is used for prediction as part of self-training.
Run pre-processing/split_data.py to execute this step.
</li>
<li>Get Pairs: This step processes the pseudo train set to format it for training. Run pre-processing/get_pairs.py to execute this step.
</li>

<li> Sentiment Pseudo Labels: This step obtains sentiment labels for the aspect-opinion pairs. Run sentiment_pseudo_labels/generate_templates.py to automatically generate templates. Next, run sentiment_pseudo_labels/template_scores.py to obtain scores for the automatically generated templates. Use these scores to rank the templates. Finally, run sentiment_pseudo_labels/evaluate_templates.py to obtain sentiment pseudo labels using highly ranked templates.

</li>


</ul>

<h3>Training</h3>
<ul>
<li>Run training/train.py to train the model.</li>
</ul>


<h2>Citations</h2>

@inproceedings{chakraborty-etal-2023-zero,
    title = "Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts",
    author = "Chakraborty, Mohna  and
      Kulkarni, Adithya  and
      Li, Qi",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.313",
    doi = "10.18653/v1/2023.acl-long.313",
    pages = "5698--5711",
    abstract = "Recent studies have demonstrated that natural-language prompts can help to leverage the knowledge learned by pre-trained language models for the binary sentence-level sentiment classification task. Specifically, these methods utilize few-shot learning settings to fine-tune the sentiment classification model using manual or automatically generated prompts. However, the performance of these methods is sensitive to the perturbations of the utilized prompts. Furthermore, these methods depend on a few labeled instances for automatic prompt generation and prompt ranking. This study aims to find high-quality prompts for the given task in a zero-shot setting. Given a base prompt, our proposed approach automatically generates multiple prompts similar to the base prompt employing positional, reasoning, and paraphrasing techniques and then ranks the prompts using a novel metric. We empirically demonstrate that the top-ranked prompts are high-quality and significantly outperform the base prompt and the prompts generated using few-shot learning for the binary sentence-level sentiment classification task.",
}

@inproceedings{10.1145/3534678.3539386,
author = {Chakraborty, Mohna and Kulkarni, Adithya and Li, Qi},
title = {Open-Domain Aspect-Opinion Co-Mining with Double-Layer Span Extraction},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539386},
doi = {10.1145/3534678.3539386},
abstract = {The aspect-opinion extraction tasks extract aspect terms and opinion terms from reviews. The supervised extraction methods achieve state-of-the-art performance but require large-scale human-annotated training data. Thus, they are restricted for open-domain tasks due to the lack of training data. This work addresses this challenge and simultaneously mines aspect terms, opinion terms, and their correspondence in a joint model. We propose an Open-Domain Aspect-Opinion Co-Mining (ODAO) method with a Double-Layer span extraction framework. Instead of acquiring human annotations, ODAO first generates weak labels for unannotated corpus by employing rules-based on universal dependency parsing. Then, ODAO utilizes this weak supervision to train a double-layer span extraction framework to extract aspect terms (ATE), opinion terms (OTE), and aspect-opinion pairs (AOPE). ODAO applies canonical correlation analysis as an early stopping indicator to avoid the model over-fitting to the noise to tackle the noisy weak supervision. ODAO applies a self-training process to gradually enrich the training data to tackle the weak supervision bias issue. We conduct extensive experiments and demonstrate the power of the proposed ODAO. The results on four benchmark datasets for aspect-opinion co-extraction and pair extraction tasks show that ODAO can achieve competitive or even better performance compared with the state-of-the-art fully supervised methods.},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {66–75},
numpages = {10},
keywords = {review analysis, natural language processing, data mining},
location = {Washington DC, USA},
series = {KDD '22}
}

</html>

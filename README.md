# MovieQA - Answering (CVPR 2016)

<strong>MovieQA: Understanding Stories in Movies through Question-Answering</strong>  
M. Tapaswi, Y. Zhu, R. Stiefelhagen, A. Torralba, R. Urtasun, and S. Fidler  
Computer Vision and Pattern Recognition (CVPR), June 2016.  
[Project page](http://movieqa.cs.toronto.edu) |
[Read the paper](http://movieqa.cs.toronto.edu/static/files/CVPR2016_MovieQA.pdf) |
[Register and download](http://movieqa.cs.toronto.edu/register/)

This repository contains approaches introduced in the above paper.

To replicate the different models and results please follow the instructions below:

---

### Add the dataset folder to path
Change the path in <code>movieqa_importer.py</code>.

----

### Encode QAs and Text
Represent the QAs and different text sources using TFIDF (vocabulary embedding), Word2Vec (word embedding), and SkipThoughts (sentence embedding).

<code>python encode_qa_and_text.py</code>

##### Prerequisities
+ Word2Vec model trained on 1364 movie plot synopses. [Download here](https://cvhci.anthropomatik.kit.edu/~mtapaswi/downloads/movie_plots_1364.d-300.mc1.w2v) and store to "models" folder
+ Skip-Thought encoder. [Github repo](https://github.com/ryankiros/skip-thoughts)
Please follow instructions on that repository.
To encode using GPU (for SkipThoughts) you may want to use
<code>THEANO_FLAGS=device=gpu python encode_qa_and_text.py</code>

----

### Hasty Student
Try to answer questions without looking at the story. This allows to analyze the bias in the dataset collection.
We evaluate different options here including answering based on:
(i) length of the answers;
(ii) within answer similarity or distinctness; and
(iii) question-answer similarity.

<code>python hasty_machine.py -h</code>

----

### Searching Student
Answer questions by searching for the best matching (set) of story sentences to the question and answer option.
We evaluate different options here such as:
(i) story sources: split_plot, subtitle, script, dvs;
(ii) representations: tfidf, word2vec, skipthought; and
(iii) window size for the story.

<code>python cosine_similarity.py -h</code>

----

### Searching Student with Convolutional Brain
Emulates searching student, but with multiple 1x1 convolutional layers that combine similarity between \<story,question\> and \<story,answer\> products.
Uses max and average pooling across sentences at the end.
We evaluate different options including:
(i) story sources: split_plot, subtitle, script, dvs; and
(ii) representations: tfidf, word2vec, skipthought.

<code>python sscb.py -h</code>  

Inputs contain an option for <code>pad_percentile</code>.
This controls the truncation or padding of zeros to stories to create batches.
The value can be in the range of 90 to 100 (default) depending on your GPU memory.

Quirks:
SSCB seems to be fairly sensitive to initialization.
We overcome this issue by training several networks (random start) and pick the model that shows best performance on the internal dev set.

----

### Modified Memory Networks
Answer questions using a modified version of the End-To-End Memory Network [arXiv](https://arxiv.org/abs/1503.08895). The modifications include use of a fixed word embedding layer along with a shared linear projection, and the ability to pick one among multiple-choice multi-word answers. The memory network supports answering in all sources. The main options to run this program are:
(i) story sources: split_plot, subtitle, script, dvs;
(ii) number of memory layers (although this did not affect performance much); and
(iii) training parameters: batch size, learning rate, #epochs.

For more details please refer to:

<code>python memory_network_text.py -h</code>

----

### Video-based Answering
Releasing code for this is fairly complicated as it comes from several projects.
Still working on updating this.

----

### Requirements (in one place)

- Word2Vec: Python installation using <code>pip install word2vec</code>
- SkipThoughts: [Github repo](https://github.com/ryankiros/skip-thoughts)
- Theano: [Github repo](https://github.com/Theano/Theano), tested on some versions of Theano-0.7 and 0.8.
- scikit-learn (PCA)
- python-gflags
- progressbar
- optparse
- nltk
- scipy
- numpy


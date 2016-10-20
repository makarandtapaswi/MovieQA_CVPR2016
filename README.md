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
coming soon

<code>python cosine_similarity.py</code>

----

### Searching Student with Convolutional Brain
coming soon

<code>python sscb.py</code>

Quirks:
This needs to be run several times. We pick the method that performs best on the internal dev set and then evaluate on the val set. Apparently the initialization of the model is very critical and shows wide variations in performance. 

----

### Modified Memory Networks
coming soon

<code>python mqa_memN2N.py</code>

----

### Video-based Answering
Still working on updating this.

----

### Requirements (in one place)

- Word2Vec: Python installation using <code>pip install word2vec</code>
- SkipThoughts: [Github repo](https://github.com/ryankiros/skip-thoughts)
- Theano: [Github repo](https://github.com/Theano/Theano), tested on some versions of Theano-0.7 and 0.8.
- python-gflags
- scipy
- numpy



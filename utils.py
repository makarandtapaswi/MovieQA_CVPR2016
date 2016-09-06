import os
import sys
import ipdb
import re
import cPickle as pkl
import numpy as np
from nltk.stem.snowball import SnowballStemmer

QA_DESC_TEMPLATE = 'descriptor_cache/qa.%s/%s.npy'  # descriptor, qid
DOC_DESC_TEMPLATE = 'descriptor_cache/%s.%s/%s.npy'  # document-type, descriptor, imdb-key
TFIDF_TEMPLATE = 'descriptor_cache/tfidf/source-%s.wthr-%d.pkl'  # document-type, word-threshold
RESULTS_TEMPLATE = 'results/%s.results'  # method.document-type.parameters

re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')
snowball = SnowballStemmer("english")


def fail_nicely(print_string, parser):
    """Fails nicely when wrong inputs are received. parser is OptionsParser()
    """

    print print_string + '\n----------------------------------\n'
    parser.print_help()
    sys.exit(1)


def restrict_to_story_type(ANS, QA):
    """Restricts the set of ground truth answer options to QAs answerable
    using this story type.
    """

    QA_qids = [qa.qid for qa in QA]
    ANS = {tak:tav for tak, tav in ANS.iteritems() if tak in QA_qids}
    return ANS


def evaluate(ANS, method, method_ans):
    """Compares predicted against ground-truth answers.
    """

    # make sure all the questions are answered
    assert sorted(method_ans.keys()) == sorted(ANS.keys()), \
                '{}, All questions in the test set not answered!' %(method)
    # mark correct
    correct = []
    for qid in ANS.keys():
        if ANS[qid] == method_ans[qid]:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = 100.0 * sum(correct) / len(ANS)
    print '%40s | acc. %.2f' %(method, accuracy)
    return correct, accuracy


def write_answerkeys_to_file(filename, answer_keys):
    """Writes answer keys to file which can be uploaded to the evaluation benchmark.
    """

    with open(filename, 'w') as fid:
        for k in answer_keys.keys():
            fid.write('%s %d\n' %(k, answer_keys[k])) 
    print "Wrote results to:", filename


def process_answer_keys(evalset, QA, ans_keys):
    """Process or evaluate answer keys.
    For validation set, compute accuracy.
    For test set, create upload formatted text files.
    """

    if evalset in ['train', 'val']:
        ### Evaluate all answer keys
        VAL_ANS = {qa.qid:int(qa.correct_index) for qa in QA}
        for method in ans_keys.keys():
            correct, acc = evaluate(VAL_ANS, method, ans_keys[method])
            results_fname = RESULTS_TEMPLATE %(method)
            with open(results_fname, 'w') as fid:
                fid.write('%.3f\n' %(acc))

    elif evalset == 'test':
        ### Write answer sets to file
        for method in ans_keys.keys():
            filename = os.path.join('upload_these', 'test.' + method + '.txt')
            write_answerkeys_to_file(filename, ans_keys[method])

    else:
        print('Unknown evaluation set!')
        sys.exit(1)


def normalize_alphanumeric(line):
    """Strip all punctuation, keep only alphanumerics.
    """

    line = re_alphanumeric.sub('', line)
    line = re_multispace.sub(' ', line)
    return line


def normalize_stemming(line):
    """Perform stemming on the words.
    """

    words = line.split(' ')
    words = [snowball.stem(word) for word in words]
    line = ' '.join(words)
    return line


def sliding_window_text(source_list, windowsize=5):
    """Convert a list of text using a sliding window (hop size = 1).
    """

    # convert a list into sliding window form
    sliding_window_list = []
    for k in range(len(source_list) - windowsize + 1):
        sliding_window_list.append(source_list[k : k+windowsize])

    for k in range(len(sliding_window_list)):
        sliding_window_list[k] = ' '.join(sliding_window_list[k])

    return sliding_window_list


def load_story_feature(imdb_key, story, feature):
    """Loads story feature for movie of imdb_key.
    """

    if feature.startswith('tfidf'):
        # TFIDF features are saved in sparse matrix format.
        with open(DOC_DESC_TEMPLATE % (story, feature, imdb_key), 'rb') as fid:
            story_features = pkl.load(fid)
        return story_features.todense()
    else:
        # Word2Vec and SkipThought features are stored as numpy arrays
        return np.load(DOC_DESC_TEMPLATE % (story, feature, imdb_key))

def load_qa_feature(qa, feature):
    """Load QA features for a particular QA.
    """

    return np.load(QA_DESC_TEMPLATE % (feature, qa.qid))


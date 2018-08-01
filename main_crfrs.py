import os
from collections import Counter, defaultdict
from random import sample, shuffle
import warnings

import numpy as np
from scipy.spatial import distance
import scipy.stats
import sklearn_crfsuite
from gensim.models import KeyedVectors
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn_crfsuite import metrics
from sklearn.preprocessing import MultiLabelBinarizer


data_path = 'data/raw/'

RESULT_FILE_csv = 'crfrs_results.csv'

def load_dataset_map(data_path):
    """ This procedure imports the dataset from the specified datapath and returns it in a dictionary, the procedure
    requires a working server of CoreNLP that needs to be properly specified in the os.environ['CORENLP_HOME'] variable
    that takes the files and properly chunks them in sentences. The output format is the
    dictionary (keyword, list of sentences from all documents)
    :param data_path: is the path to find data to load, is expected to be a folder containing a subfolder feature and label
        :type data_path: str
    :return data: contaning the list of sentences from all documents associated with a key
        :rtype data: defaultdict
    """
    data = defaultdict()
    for root, _, files in os.walk(data_path + 'feature/'):
        for name in files:

            feature_path = os.path.join(root, name)
            label_path = feature_path.replace('feature', 'label')

            m_key = set(sum([ x.lower().replace('*', '').split(', ') for x in open(label_path).read().splitlines() ], []))

            data[ tuple([label_path]) + tuple(m_key) ] = feature_path.replace('label', 'feature')

    return data


def select_keywords(data, subsample=None, target_count=None, min_count=None, max_count=None, composed_words=None, disjoint=None):
    """Checks in the KEY_OCCURRENCES which keyword have a document population respecting required format
    :param data:
    :param subsample: how many keywords I want to select
    :param target_count: how many keywords I want to select
    :param min_count: minimum amount of documents of a keyword to be selected
    :param max_count: maximum amount of documents of a keyword to be selected
    :param composed_words: either or not the kesy accepted can be in the format of class/word/ or multiple words
    :param disjoint: either or not the document selected need to be disjoint between different keys
        :type data: dict
        :type subsample: int
        :type target_count: int
        :type min_count: int
        :type max_count: int
        :type composed_words: bool
        :type disjoint: bool
    :return: data
    :return: keys
        :rtype data: dict
        :rtype keys: set
    """

    def remove_non_target_keys(keys, target_keys):
        """ Remove keys that are not target from dataset
        :param keys:
        :param target_keys:
            :type keys: tuple
            :type target_keys: set
        :return:
            :rtype : tuple
        """
        return (keys[0],) + filter(lambda x : x in target_keys, keys[1:])

    key_occurrences = Counter( [ j for i in [ list(k[1:]) for k, _ in data.items() ] for j in i ])

    if min_count is not None:
        key_occurrences = { k : v for k, v in key_occurrences.items() if v >= min_count}

    if max_count is not None:
        key_occurrences = { k : v for k, v in key_occurrences.items() if v <= max_count}

    if not composed_words :
        key_occurrences = { k : v for k, v in key_occurrences.items() if ' ' not in k and '/' not in k }

    if min_count is not None:
        key_occurrences = { k : v for k, v in key_occurrences.items() if int(v) >= min_count }

    if disjoint :
        data = filter(lambda k : bool(len(set(k[1:]).intersection(set(key_occurrences.keys()))) == 1), data.keys())
        key_occurrences = Counter(sum([ k[ 1: ] for k, _ in data.items() ], [ ]))

    data = { k : v for k, v in data.items() if bool(not set(k[1:]).isdisjoint(set(key_occurrences.keys()))) }

    if subsample is not None and target_count is not None:
        docs_per_key = subsample / target_count
        target_keys = sample([ k for k, v in key_occurrences.items() if v >= docs_per_key ], target_count)
        output_data = defaultdict(list)
        for k, v in data.items():
            if not set(k[ 1: ]).isdisjoint(set(target_keys)):
                for d in set(k[ 1: ]).intersection(set(target_keys)):
                    if len(output_data[ d ]) <= docs_per_key:
                        output_data[ d ] += [ { k: v } ]

        data = { k : v for d_list in output_data.values() for d in d_list for k, v in d.items() }

    target_concepts = set(key_occurrences.keys()).intersection( set(sum([ list(k[1:]) for k in data.keys() ], [])) )

    data = { remove_non_target_keys(k, target_concepts) : v for k, v in data.items() }

    return data, target_concepts


def load_sentences(dataset):
    """ loads the dataset as the value of the map instea of the path
    :param dataset:
        :type dataset: dict
    :return:
        :rtype: dict
    """
    return { k : open(v, 'r').readlines() for k, v in dataset.items() }


def split_dataset(data, keys, p=[ .6, .2, .2 ]):
    """ This procedure splits a dataset, with split_percentage, and filters it keeping only specified keys,
    :param data: dictionary with all datat in (key, sentences) format
    :param keys: list of required keys
    :param p: percentage of the test_set set, what is not in test_set set is in train_set set
        :type data: dict
        :type keys: set
        :type p: list
    :return: train_set
    :return: test_set
    :return: eval_set
        :rtype train_set: dict
        :rtype test_set: dict
        :rtype eval_set: dict
    """
    train_set = dict()
    test_set = dict()
    eval_set = dict()

    for key in keys:
        files = list({ k[0] for k, v in data.items() if not key in k })
        shuffle(files)

        pin = [ int(p[ 0 ] * len(files)), int(p[ 0 ] * len(files) + p[ 1 ] * len(files)) ]

        train = files[ : pin[ 0 ] ]
        test = files[ pin[ 0 ]: pin[ 1 ] ]
        eval = files[ pin[ 1 ]: ]

        train_set.update({ k : v for k, v in data.items() if k[ 0 ] in train })
        test_set.update({ k : v for k, v in data.items() if k[ 0 ] in test })
        eval_set.update({ k : v for k, v in data.items() if k[ 0 ] in eval })

    print 'train n test / train u test ==>' , len(
        set(sum([ list(k[1:]) for k, v in train_set.items() ], [])).intersection(set(sum([ list(k[1:]) for k, v in test_set.items() ], [])))
    ) / (float(len(
        set(sum([ list(k[1:]) for k, v in train_set.items() ], [])).union(set(sum([ list(k[1:]) for k, v in test_set.items() ], [])))
    )) + .00000001)
    print 'train n eval / train u eval ==>' , len(
        set(sum([ list(k[1:]) for k, v in train_set.items() ], [])).intersection(set(sum([ list(k[1:]) for k, v in eval_set.items() ], [])))
    ) / (float(len(
        set(sum([ list(k[1:]) for k, v in train_set.items() ], [])).union(set(sum([ list(k[1:]) for k, v in eval_set.items() ], [])))
    )) + 0.0000001)
    print 'if bigger than 1 those are 0'

    return train_set, test_set, eval_set


def load_embeddings(embedding_layer='512', epochs='15', limit=None):
    """ Loads the embeddings in the gensim keyedVectors format from the specified pretrained file
    :param embedding_layer: embedding layer dimension
    :param epochs: number of train epochs
    :param limit: max number of loaded words
        :type embedding_layer: str
        :type epochs: str
        :type limit: int
    :return: model
        :rtype model: KeyedVectors
    """
    return KeyedVectors.load_word2vec_format(
        'embeddings/lay_%s/epo_%s/vectors.txt' % (embedding_layer, epochs),
        binary=False,
        limit=limit
    )


def format_input_crf(data, model=None, distance_threshold=None, window=None):
    """ This procedure takes in input the train and test set and then annotates with iob notation with the specified
    wordToVec model, window and threshold
    :param data: the data dictionary with keys, list of sentences
    :param model: the trained wordToVec model
    :param distance_threshold: the maximum distance in word2vec space to consider a word part of his key concept
    :param window: the size of the span that will be annotated with the I-tag if a word in positive to a concept
        :type data: dict
        :type model: KeyedVectors
        :type distance_threshold: float
        :type window: int
    :return m_data:
        :rtype m_data: list of tuple
    """

    def annotate_set(data, model, distance_threshold=.2, window=3):
        """ This procedure is getting a list of sentences and theyr overall annotation and it will compute and output its
        full IOB-annotation
        :param dataset:
        :param data: dictionary containing the overall annotation and list of sentences
        :param model: pretrained word2vec model with
        :param distance_threshold: max distance allowed in word2vec space to consider a word as referring to a concept
        :param window: size of the span that will be annotated with the given concept when a word is within distance_threshold
            :type dataset: dict
            :type data: dict
            :type model: KeyedVectors
            :type distance_threshold: float
            :type window: int
        :return: data_iob
            :rtype data_iob: defaultdict
        """

        def word_to_vec(model, word=''):
            """ This procedure returns the relative vector in word2vec space of the given word, if the word is composite return
            the medium point of the found ones. If model does not contain the word returns None
            :param model: word2vec pretrained model
            :param words: word to be translated into point in word2vec space
                :type model: KeyedVectors
                :type words: str
            :return: retval
                :rtype retval: list
            """
            try:
                retval = np.array([ model[ word ] ], dtype=float)
            except KeyError:
                retval = None

            return retval

        def sentence_to_vec(model, sentence=''):
            """ This procedure translates a list of words in a list of points in word2vec space
            :param model: word2vec pretrained model
            :param sentence: list of words
                :type model: KeyedVectors
                :type sentence: str
            :return: list of points
                :rtype: list
            """
            return [ word_to_vec(word=word, model=model) for word in sentence.split() ]

        def annotate(vec, concepts_embedding, distance_threshold=.2):
            """ This procedure decided either a word in within minimum distance from a concept or not
            :param vec: word representation in word2vec space
            :param concepts_embedding: concept representation in word2vec space
            :param distance_threshold: max distance allowed from concept to consider work represented by the concept
                :type vec: list
                :type concepts_embedding: dict
                :type distance_threshold: float
            :return: 'I' or 'O'
                :rtype: str
            """
            if vec is not None:
                for k, concept_embedding in concepts_embedding.items():
                    if concept_embedding is not None and distance.euclidean(vec,
                                                                            concept_embedding) < distance_threshold:
                        return 'I-' + k
            return 'O'

        def propagate_iob(values, concept):
            """Given a concept and the subset to annotate it returns the subset proper annotation
            :param values : list of annotation
            :param concept : concept to propagate over the list
                :type values: list
                :type concept: str
            :return : (values, annotation)
                :rtype : tuple
            """
            try :
                first_value = next(values.index(x) for x in values if x == 'O')
                values[ first_value ] = 'B-' + concept
                for value in values[ first_value: ]:
                    if value == 'O':
                        values[ values.index(value) ] = 'I-' + concept
            except:
                pass
            return values

        concept_embeddings = { c : word_to_vec(model=model, word=c) for c in set(sum([ list(k[1:]) for k in data.keys() ],[])) }

        data_zipped = []
        for concepts, sentences in data.items():
            concepts_e = { c : e for c, e in concept_embeddings.items() if c in concepts }
            for sentence in sentences:
                sentence_vec = sentence_to_vec(sentence=sentence, model=model)
                sentence_iob = [ annotate(vec=vec, concepts_embedding=concepts_e, distance_threshold=distance_threshold) for vec in sentence_vec ]

                for position in range(len(sentence_iob)):
                    if sentence_iob[ position ].startswith('I') and (
                                    sentence_iob[ position - 1 ] is None or sentence_iob[ position - 1 ] == 'O'):
                        if position - window < 0:
                            sentence_iob[ 0: position + window + 1 ] = propagate_iob(
                                values=sentence_iob[ 0: position + window + 1 ], concept=sentence_iob[ position ][ 2: ])
                        elif position + window > len(sentence_iob):
                            sentence_iob[ position - window: len(sentence_iob) ] = propagate_iob(
                                values=sentence_iob[ position - window: len(sentence_iob) ],
                                concept=sentence_iob[ position ][ 2: ])
                        else:
                            sentence_iob[ position - window: position + window + 1 ] = propagate_iob(
                                values=sentence_iob[ position - window: position + window + 1 ],
                                concept=sentence_iob[ position ][ 2: ])

                data_zipped += [ (sentence, sentence_iob) ]

        return data_zipped

    data_iob = None
    if model and distance_threshold and window :
        data_iob = annotate_set(data=data, model=model, distance_threshold=distance_threshold, window=window)

    m_data = []
    if data_iob is not None:
        for sentence, annotation in data_iob:
            m_data += [ [ (sentence.split()[ cursor ], annotation[ cursor ]) for cursor in range(len(annotation)) ] ]
    else:
        for sentences in data.values():
            for sentence in sentences:
                m_data += [ [ (word, None) for word in sentence.split() ] ]
    return m_data


def get_Xy(sents):
    """Create input for the crf with template
    :param sents:
        :type sents: list
    :return:
        :rtype : list
        :rtype : list
    """
    def sent2features(sent):
        """
        :param sent:
            :type sent: list
        :return: list
        """

        def word2features(sent, i):
            """
            :param sent:
            :param i:
                :type sent: list
                :type i: int
            :return:
                :rtype : dict
            """
            word = sent[ i ][ 0 ]

            features = {}

            try:
                features.update({'word[:+3]': word[ :+3 ]})
            except:
                pass
            try:
                features.update({'word-5': sent[ i - 5 ][ 0 ]})
            except:
                pass
            try:
                features.update({'word-4': sent[ i - 4 ][ 0 ]})
            except:
                pass
            try:
                features.update({'word-3': sent[ i - 3 ][ 0 ]})
            except:
                pass
            try:
                features.update({ 'word-2': sent[ i - 2 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word-1': sent[ i - 1 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word': word })
            except:
                pass
            try:
                features.update({ 'word+1': sent[ i + 1 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word+2': sent[ i + 2 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word+3': sent[ i + 3 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word+4': sent[ i + 4 ][ 0 ] })
            except:
                pass
            try:
                features.update({ 'word+5': sent[ i + 5 ][ 0 ] })
            except:
                pass

            return features

        return [ word2features(sent, i) for i in range(len(sent)) ]

    def sent2labels(sent):
        """
        :param sent:
            :type sent: list
        :return:
            :rtype : list
        """
        return [ label for _, label in sent ]

    return [sent2features(s) for s in sents], [sent2labels(s) for s in sents]


def fit_CRF_model(X, y, algorithm='l2sgd', c1=0.1, c2=1.5, max_iterations=200, all_possible_transitions=True, verbose=False):
    """ Returns fit crf model
    :param X:
    :param y:
    :param algorithm:
    :param c1:
    :param c2:
    :param max_iterations:
    :param all_possible_transitions:
        :type X: list
        :type y: list
        :type algorithm: str
        :type c1: float
        :type c2: float
        :type max_iterations: int
        :type all_possible_transitions: bool
    :return: fit model
        :rtype : sklearn_crfsuite.CRF
    """
    crf = sklearn_crfsuite.CRF(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions,
        verbose=verbose
    )
    crf.fit(X, y)
    return crf


def get_labels(crf):
    """ Returns all possible labes of the model (without 'O')
    :param crf:
        :type crf: sklearn_crfsuite.CRF
    :return labels:
        :rtype labels: list
    """
    labels = list(crf.classes_)
    labels.remove('O')
    return labels


def sort_labels(labels):
    """Guess what this Sorts the labels
    :param labels:
        :type labels: list
    :return :
        :rtype : list
    """
    return sorted(labels, key=lambda name: (name[1:], name[0]))


def print_results(labels, y_test, y_pred):
    """ Prints metrics
    :param labels:
    :param y_test:
    :param y_pred:
        :type labels: list
        :type y_test: list
        :type y_pred: list
    """
    sorted_labels = sort_labels(labels)
    print(metrics.flat_classification_report( y_test, y_pred, labels=sorted_labels, digits=3))


def random_search(X, y, labels, params_space, average='weighted', algorithm='l2sgd', max_iterations=200, all_possible_transitions=True, cv=3, verbose=2, n_jobs=-1, n_iter=2):
    """ Random search best model
    :param X:
    :param y:
    :param labels:
    :param params_space:
    :param average:
    :param algorithm:
    :param max_iterations:
    :param all_possible_transitions:
    :param cv:
    :param verbose:
    :param n_jobs:
    :param n_iter:
        :type X: list
        :type y: list
        :type labels: list
        :type params_space: dict
        :type average: str
        :type algorithm: str
        :type max_iterations: int
        :type all_possible_transitions: bool
        :type cv: int
        :type verbose: int
        :type n_jobs: int
        :type n_iter: int
    :return :
        :rtype : sklearn_crfsuite.CRF
    """
    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm=algorithm,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions
    )

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average=average, labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv,
                            verbose=verbose,
                            n_jobs=n_jobs,
                            n_iter=n_iter,
                            scoring=f1_scorer)
    rs.fit(X, y)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    return rs.best_estimator_


def print_report_RS(model, y_test, y_pred, sorted_labels):
    """
    :param model:
    :param y_test:
    :param y_pred:
    :param sorted_labels:
        :type model: sklearn_crfsuite.CRF
        :type y_test: list
        :type y_pred: list
        :type sorted_labels: list
    """
    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

    print("Top likely transitions:")
    print_transitions(Counter(model.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(model.transition_features_).most_common()[-20:])

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(model.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(model.state_features_).most_common()[-30:])


def evaluate_solution(model, eval_set, target_keys):
    """ This procedure gives the score of the document labeling
        :param model:
        :param eval_set:
        :param target_keys:
            :type model:
            :type eval_set: dict
            :type target_keys: set
        :return
    """

    def label_document(document, document_path):
        """ The document inherits from the text labeling the labels
            :param document:
            :param document_path:
                :type document: dict
                :type document_path: str
            :return: ( f1_score macro, f1_score micro, f1_score weighted, accuracy_score normalize=True,
               precision_score micro, precision_score macro, precision_score weighted, recall_score micro,
               recall_score macro, recall_score weighted )
                :rtype: tuple
        """
        eval_sents = format_input_crf(data=document)
        X_eval, _ = get_Xy(eval_sents)
        y_pred = model.predict(X_eval)

        return { document_path: { w[2:] for s in y_pred for w in s if w is not 'O' } }

    labeling = defaultdict()
    for concept, sentences in eval_set.items():
        eval_subset = { concept[1:] : sentences }

        labeling.update( label_document(document=eval_subset, document_path=concept[0]) )

    labeling = { k : filter(lambda x : x in target_keys, v) for k, v in labeling.items() }

    paths = { path[ 0 ] for path in labeling.items() }

    y_pred = [ sorted(labeling[ path ]) for path in paths ]

    paths_labels = { path : open(path, 'r').read().splitlines() for path in paths }
    paths_labels_cleaned = { k : map(str.lower, v) for k, v in paths_labels.items() }
    gold = { k : filter(lambda x: x in target_keys, v) for k, v in paths_labels_cleaned.items() }

    y_true = [ sorted(gold[path]) for path in paths ]

    binarizer = MultiLabelBinarizer()

    binarizer.fit(np.array([list(target_keys)]))

    return  f1_score(binarizer.transform(y_true), binarizer.transform(y_pred), average='macro'), \
            f1_score(binarizer.transform(y_true), binarizer.transform(y_pred), average='micro'), \
            f1_score(binarizer.transform(y_true), binarizer.transform(y_pred), average='weighted'), \
            accuracy_score(binarizer.transform(y_true), binarizer.transform(y_pred), normalize=True), \
            precision_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'micro'), \
            precision_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'macro'), \
            precision_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'weighted'), \
            recall_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'micro'),\
            recall_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'macro'),\
            recall_score(binarizer.transform(y_true), binarizer.transform(y_pred), average= 'weighted')#,\

def main(
    embedding_layer = '512',
    epochs = '15',
    words_loaded = 50000,
    labeling_threshold = 4,
    window_size = 2,

    subsample = 10,
    target_count = 10,
    min_count = 1000,
    max_count = 10000,
    composed_words = False,
    disjoint = False,
    train_test_evaluate = [ .6, .2, .2 ],

    algorithm='l2sgd',
    c1=0.1,
    c2=1.5,
    max_iterations=200,
    all_possible_transitions=True,
    scale = [.8, 2.5],
    cv = 3,
    verbose_rs = 2,
    verbose_bs = True,
    n_jobs = -1,
    n_iter = 2,
    average = 'weighted'
):
    """ Import, preprocess, label, train, test, random search, evaluate baseline and advanced algs
    :param embedding_layer:
    :param epochs:
    :param words_loaded:
    :param labeling_threshold:
    :param window_size:
    :param subsample:
    :param target_count:
    :param min_count:
    :param max_count:
    :param composed_words:
    :param disjoint:
    :param train_test_evaluate:
    :param algorithm:
    :param c1:
    :param c2:
    :param max_iterations:
    :param all_possible_transitions:
    :param scale:
    :param cv:
    :param verbose_rs:
    :param verbose_bs:
    :param n_jobs:
    :param n_iter:
    :param average:
        :type embedding_layer: str
        :type epochs: str
        :type words_loaded: int
        :type labeling_threshold: float
        :type window_size: int
        :type subsample: int
        :type target_count: int
        :type min_count: int
        :type max_count: int
        :type composed_words: bool
        :type disjoint: bool
        :type train_test_evaluate: list
        :type algorithm: str
        :type c1: float
        :type c2: foat
        :type max_iterations: int
        :type all_possible_transitions: bool
        :type scale: list of float
        :type cv: int
        :type verbose_rs: int
        :type verbose_bs: bool
        :type n_jobs: int
        :type n_iter: int
        :type average: str
    :return: ( embedding_layer, epochs, words_loaded, labeling_threshold, window_size, subsample, target_count,
               min_count, max_count, composed_words, disjoint, train_test_evaluate, algorithm, c1, c2,
               max_iterations, all_possible_transitions, scale, cv, verbose_rs, verbose_bs, n_jobs, n_iter,
               average, f1_score macro, f1_score micro, f1_score weighted, accuracy_score normalize=True,
               precision_score micro, precision_score macro, precision_score weighted, recall_score micro,
               recall_score macro, recall_score weighted)
        :rtype: str
    """
    print '\n--load dataset representation'
    dataset = load_dataset_map(data_path=data_path)
    dataset, target_concepts = select_keywords(dataset, subsample=subsample, target_count=target_count, min_count=min_count, max_count=max_count, composed_words=composed_words, disjoint=disjoint)
    print '--done'

    print '\n--split data and format to meet crf reqs'
    train_set, test_set, eval_set = split_dataset(data=dataset, keys=target_concepts, p=train_test_evaluate)
    print '--done'

    print '\n--load embedding model and format train and test set for CRF'
    model = load_embeddings(embedding_layer=embedding_layer, epochs=epochs, limit=words_loaded)
    train_set = load_sentences(train_set)
    train_sents = format_input_crf(data=train_set, distance_threshold=labeling_threshold, model=model, window=window_size)
    test_set = load_sentences(test_set)
    test_sents = format_input_crf(data=test_set, distance_threshold=labeling_threshold, model=model, window=window_size)
    print '--done'

    print '\n--get train and test tuples'
    X_train, y_train = get_Xy(train_sents)
    X_test, y_test = get_Xy(test_sents)
    print '--done'

    print '\n--fit crf (BASELINE)'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        baseline = fit_CRF_model(X=X_train, y=y_train, algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=all_possible_transitions, verbose=verbose_bs)
    labels = get_labels(crf=baseline)
    y_pred = baseline.predict(X_test)
    print_results(labels=labels, y_test=y_test, y_pred=y_pred)
    print '--done'

    print '\n--randomized search (CHAMPION)'
    params_space = { 'c2': scipy.stats.uniform(scale[0], scale[1]) }
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        champion = random_search(X=X_train, y=y_train, labels=labels, params_space=params_space, average=average, algorithm=algorithm, max_iterations=max_iterations, all_possible_transitions=all_possible_transitions, cv=cv, verbose=verbose_rs, n_jobs=n_jobs, n_iter=n_iter)
    print_report_RS(champion, y_test=y_test, y_pred=champion.predict(X_test), sorted_labels=sort_labels(labels))
    print '--done'

    print '\n--evaluation (CHAMPION)'
    eval_set = load_sentences(eval_set)
    evals = evaluate_solution(model=champion, eval_set=eval_set, target_keys=target_concepts)
    print '--done'

    retval = [ ]
    retval += [ str(embedding_layer) ]
    retval += [ str(epochs) ]
    retval += [ str(words_loaded) ]
    retval += [ str(labeling_threshold) ]
    retval += [ str(window_size) ]
    retval += [ str(subsample) ]
    retval += [ str(target_count) ]
    retval += [ str(min_count) ]
    retval += [ str(max_count) ]
    retval += [ str(composed_words) ]
    retval += [ str(disjoint) ]
    retval += [ '"' + str(train_test_evaluate) + '"' ]
    retval += [ str(algorithm) ]
    retval += [ str(c1) ]
    retval += [ str(c2) ]
    retval += [ str(max_iterations) ]
    retval += [ str(all_possible_transitions) ]
    retval += [ '"' + str(scale) + '"' ]
    retval += [ str(cv) ]
    retval += [ str(n_jobs) ]
    retval += [ str(n_iter) ]
    retval += [ str(average) ]
    retval += [ str(v) for v in evals ]

    return ', '.join(retval)

os.system( 'echo "\n" >> %s' % RESULT_FILE_csv)
os.system( 'echo "\n" >> %s' % RESULT_FILE_csv)
os.system( 'echo "NEW BULK" >> %s' % RESULT_FILE_csv)

header = []
header += [ 'embedding_layer' ]
header += [ 'epochs' ]
header += [ 'words_loaded' ]
header += [ 'labeling_threshold' ]
header += [ 'window_size ' ]
header += [ 'subsample' ]
header += [ 'target_count' ]
header += [ 'min_count' ]
header += [ 'max_count' ]
header += [ 'composed_words' ]
header += [ 'disjoint' ]
header += [ 'train_test_evaluate' ]
header += [ 'algorithm' ]
header += [ 'c1' ]
header += [ 'c2' ]
header += [ 'max_iterations' ]
header += [ 'all_possible_transitions' ]
header += [ 'scale' ]
header += [ 'cv' ]
header += [ 'n_jobs' ]
header += [ 'n_iter' ]
header += [ 'average' ]
header += [ 'f1_score macro' ]
header += [ 'f1_score micro' ]
header += [ 'f1_score weighted' ]
header += [ 'accuracy_score normalize=True' ]
header += [ 'precision_score micro' ]
header += [ 'precision_score macro' ]
header += [ 'precision_score weighted' ]
header += [ 'recall_score micro' ]
header += [ 'recall_score macro' ]
header += [ 'recall_score weighted' ]
os.system( 'echo "%s" >> %s' % (', '.join(header) + '\n', RESULT_FILE_csv))

os.system('echo "%s" >> %s' % (
        main(embedding_layer='512', epochs='15', words_loaded=50000, labeling_threshold=4.5, window_size=2, subsample=None,
             target_count=7, min_count=None, max_count=None, composed_words=None, disjoint=None,
             train_test_evaluate=[ .6, .2, .2 ], algorithm='l2sgd', c1=None, c2=1.5, max_iterations=200,
             all_possible_transitions=True, scale=[.8, 2.5], cv=3, verbose_rs=2, verbose_bs=True, n_jobs=-1, n_iter=3,
             average='weighted'
        ), RESULT_FILE_csv)
)


print '\n\n--DONE'


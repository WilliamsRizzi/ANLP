import hashlib
import os
from collections import defaultdict, Counter
from random import randint, shuffle
from random import sample

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from gensim.models import KeyedVectors
from numpy.random import choice
from scipy.spatial import distance
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

THREADS_LEARN = '15'

data_path = 'data/raw/'

TEMPLATES = 'templates'
MODELS = 'models'
SANDBOX = 'sandbox'
INITIAL_POPULATION_FOLDER = 'initial_population'
DATA = 'data'

CHAMPION_MODEL = 'champion_model'

train_file = 'train.data'
test_file = 'test.data'
eval_file = 'eval.data'

RESULT_FILE_csv = 'crfpp_results.csv'


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


def format_input_crf(data, destination_file, model=None, distance_threshold=None, window=None):
    """ This procedure takes in input the train and test set and then annotates with iob notation with the specified
    wordToVec model, window and threshold
    :param data: the data dictionary with keys, list of sentences
    :param destination_file:
    :param model: the trained wordToVec model
    :param distance_threshold: the maximum distance in word2vec space to consider a word part of his key concept
    :param window: the size of the span that will be annotated with the I-tag if a word in positive to a concept
        :type data: dict
        :type destination_file: str
        :type model: KeyedVectors
        :type distance_threshold: float
        :type window: int
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

    with open('%s/%s' % (DATA, destination_file), 'w+') as f:
        if data_iob is not None:
            for sentence, annotation in data_iob:
                [ f.write(sentence.split()[ cursor ] + '\t' + annotation[ cursor ] + '\n') for cursor in range(len(annotation)) ]
                f.write('\n')
        else:
            for sentences in data.values():
                for sentence in sentences:
                    [ f.write(word + '\n') for word in sentence.split() ]
                    f.write('\n')
        f.write('\n')


def genetic_feature_discovery(generations=1, tournament_size=4, selected_from_tournament=2, how_many_champions=1):
    """ Generate and train a population of CRF++ templates to solve the concept tagging task.
    This script will use a genetic algorithm to generate and select the best features
    in order to maximize the performances of a CRF++ classifier on the concept tagging task.
    We use the Python DEAP library: https://github.com/DEAP/deap.
        :parameter generations:
        :parameter tournament_size:
        :parameter selected_from_tournament:
        :parameter how_many_champions:
            :type generations: int
            :type tournament_size: int
            :type selected_from_tournament: int
            :type how_many_champions: int
        :return champ:
            :rtype champ: tools.HallOfFame
    """

    def get_id(individual):
        """ Generate a unique identifier for a set of features.
        This is done by sorting the features in alphabetically order,
        inserting a separator between them and computing a MD5 hash.
        This can be used to cache already done computations.
        :param individual: Population individual, representing a CRF++ template.
            :type individual: list #TODO
        :return: Unique ID for the individual.
            :rtype: str
        """
        return hashlib.md5('#'.join(sorted(individual)).encode('utf-8')).hexdigest()

    def register_toolbox(selected_from_tournament=2, tournament_size=4):
        """ Register toolbox with base parameters
        :parameter selected_from_tournament:
        :parameter tournament_size:
            :type selected_from_tournament: int
            :type tournament_size: int
        :return toolbox:
            :rtype toolbox: base.Toolbox()
        """

        def load_population_guess(create_population, create_individual, directory):
            """ Load the guess initial population from a group of CRF++ templates.
            This function iterates over all files in the target directory,
            loads all CRF++ templates found, remove the B feature (automatically added later)
            and convert each list of feature in a DEAP individual.
            :param create_population: Function used to convert a data structure to a DEAP population.
            :param create_individual: Function used to convert a data structure to a DEAP individual.
            :param directory: Directory where to find the CRF++ templates.
            :return: Population of individuals generated from the CRF++ templates.
            """

            population = [ ]

            for filename in os.listdir(directory):
                features = [ l.strip().split(':')[ 1 ] for l in open('%s/%s' % (directory, filename), 'r').readlines()
                             if l.strip().startswith('U') ]

                individual = create_individual(features)
                population.append(individual)

            return create_population(population)

        def random_feature():
            """Generate a random feature for the CRF++ template.
                :return: joined_feats
                    :rtype: str
            """
            row_window = 7
            col_window = 0

            order = choice([ 1, 2, 3 ], p=[ .3, .5, .2 ])
            features = [ '%x[{:d},{:d}]'.format(randint(-row_window, row_window), col_window) for _ in range(order) ]

            separators = [ '.', ',', "'", '/', '>', '<', '(', ')', '*', '&', '^', '$', '#', '@', '!', ':', ';', '|',
                           '{', '}' ]
            return choice(separators).join(features)

        def mutate(individual, toolbox, n=1):
            """Generate a new individual starting from the given one and applying some random mutations.
            A mutation can be one of the following:
              - deletion of a random feature (to get rid of the redundant / not useful ones)
              - insertion of a new feature
              - substitution of a random feature with a newly generated one
            :param individual: Original individual to mutate. NB: the original individual is NOT modified,
                               instead it is cloned a the new one is modified.
            :param toolbox: DEAP toolbox, used to clone the individual.
            :param n: Upper bound to the number of mutations to apply. The real number is randomly chosen in [1, n].
            :return: A new individual, result of the mutation of the old one.
                :rtype: tuple
            """

            def insertion(mutant):
                return mutant.append(random_feature())

            def deletion(mutant):
                return mutant.pop(randint(0, len(mutant) - 1))

            def substitution(mutant):
                mutant.pop(randint(0, len(mutant) - 1))
                return mutant.append(random_feature())

            # make a copy of the individual
            mutant = toolbox.clone(individual)
            del mutant.fitness.values

            for i in range(randint(1, n)):
                mutant = choice([ insertion, deletion, substitution ])(mutant)

            # NB: this must be a tuple
            return (mutant,)

        def test_genetic(individual):
            """ Compute the fitness function for the given individual.
            It materialized the CRF++ template, train it and measure the performances on the test set.
            This function makes use of the `run.sh` BASH script in this folder and CRF++ binaries installed on the system.
            :param individual: Population individual, representing a CRF++ template.
            :return: F1 value on the test set, as a float.
            """

            def train_crf(cut_off='5', hyperparameter_crf='1.5', alg='CRF-L2', template_file='templates/miscellaneous/m1', train_file='data/train.data', model_file='results/performances.txt'):
                """ Train the crf with the selected parameters on the TRAIN_SET, with the TEMPLATE_FILE, regards the crf parameters
                   please refer to http://taku910.github.io/crfpp/
                   :param threads: number of thread used by the train procedure
                   :param hyperparameter_crf: set the hyperparameter of the crf, for larger values the crf tends to overfit
                   :param cut_off: set the minimum amount of occourrences for the features to be took into consideration
                   :param alg: changes the regularisation algorithm choose between: ['MIRA', 'CRF-L1', 'CRF-L2']
                   :param template_file:
                   :param train_file:
                   :param model_file:
                       :type threads: str
                       :type hyperparameter_crf: str
                       :type cut_off: str
                       :type alg: str
                       :type template_file: str
                       :type train_file: str
                       :type model_file: str
                   """
                os.system('%scrf_learn -p %s -c %s -f %s -a %s %s %s %s' % (
                CRF_LIB, THREADS_LEARN, hyperparameter_crf, cut_off, alg, template_file, train_file, model_file))

            _id = get_id(individual)

            if not os.path.exists(TEMPLATES):
                os.makedirs(TEMPLATES)

            if not os.path.exists(MODELS):
                os.makedirs(MODELS)

            if not os.path.exists(SANDBOX):
                os.makedirs(SANDBOX)

            # check if I need to train this model
            performances_file = '%s/%s/performances' % (SANDBOX, _id)
            if not os.path.exists('%s/%s' % (MODELS, _id)):
                # write the model, if needed
                with open('%s/%s' % (TEMPLATES, _id), 'w') as f:
                    [ f.write('U{:03d}:{}\n'.format(i, line)) for i, line in enumerate(individual) ]
                    f.write('\nB\n')

                train_crf(
                    cut_off='5',
                    hyperparameter_crf='1.5',
                    alg='CRF-L2',
                    train_file='%s/%s' % (DATA, train_file),
                    template_file='%s/%s' % (TEMPLATES, _id),
                    model_file='%s/%s' % (MODELS, _id)
                )
                os.mkdir('%s/%s' % (SANDBOX, _id))
                test_crf(
                    model_file='%s/%s' % (MODELS, _id),
                    feature_file='%s/%s' % (DATA, test_file),
                    encoded_test='%s/%s' % (SANDBOX, 'encoded_test'),
                    result_file='%s/%s/%s' % (SANDBOX, _id, 'performances')
                )

            result = float(open(performances_file).readlines()[ 1 ].split('FB1:')[ 1 ].strip())
            print _id, '-> F1:', result

            # NB: this must be a tuple
            return (result,)

        # the individual is simply a list of features...
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register('random_feature', random_feature)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.random_feature, n=15)

        toolbox.register('evaluate', test_genetic)

        toolbox.register('mutate', mutate, toolbox=toolbox, n=3)
        toolbox.register('mate', tools.cxUniform, indpb=0.33)
        toolbox.register('select', tools.selTournament, k=selected_from_tournament, tournsize=tournament_size)

        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('population_guess', load_population_guess, list, creator.Individual, INITIAL_POPULATION_FOLDER)

        return toolbox

    toolbox = register_toolbox(selected_from_tournament=selected_from_tournament, tournament_size=tournament_size)

    # genetic algorithm: init the population and add random individuals, eaSimple evolution (mutate, select, loop)
    pop = toolbox.population_guess() + toolbox.population(n=6)
    champ = tools.HallOfFame(maxsize=how_many_champions)
    algorithms.eaSimple(pop, toolbox, halloffame=champ, cxpb=0.3, mutpb=0.5, ngen=generations)

    champ = champ[0]

    os.system('mv %s/%s %s/champion_model' % (MODELS, get_id(champ), CHAMPION_MODEL))
    os.system('mv %s/%s %s/champion_template' % (TEMPLATES, get_id(champ), CHAMPION_MODEL))

    return champ


def test_crf(model_file, feature_file, encoded_test, result_file=None):
    """ Test the crf model in the MODEL_FILE with the TEST_FILE_CRF file and writes out the result in the ENCODED_TEST
        :param model_file:
        :param feature_file:
        :param encoded_test:
        :param result_file:
            :type model_file: str
            :type feature_file: str
            :type encoded_test: str
            :type result_file: str
    """

    def label_crf(model_file, feature_file, encoded_test):
        """ Test the crf model in the MODEL_FILE with the TEST_FILE_CRF file and writes out the result in the ENCODED_TEST
            :param model_file:
            :param feature_file:
            :param encoded_test:
                :type model_file: str
                :type feature_file: str
                :type encoded_test: str
        """
        os.system('%scrf_test -m %s %s -o %s' % (CRF_LIB, model_file, feature_file, encoded_test))

    label_crf(model_file, feature_file, encoded_test)
    if result_file is not None:
        os.system('./conlleval.pl -d \'\t\' < %s >> %s' % (encoded_test, result_file))


def evaluate_solution(eval_set, target_keys):
    """ This procedure gives the score of the document labeling
        :param eval_set:
        :param target_keys:
            :type eval_set: dict
            :type target_keys: set
    """

    def label_document(document, document_path):
        """ The document inherits from the text labeling the labels
            :param document:
            :param document_path:
                :type document: dict
                :type document_path: str
            :return:
                :rtype: dict
        """
        format_input_crf(data=document, destination_file=eval_file)
        test_crf(
            model_file='%s/%s' % (CHAMPION_MODEL, 'champion_model'),
            feature_file='%s/%s' % (DATA, eval_file),
            encoded_test='%s/%s' % (CHAMPION_MODEL, 'labels')
        )

        labels = { sr.split('\t')[ 1 ][ 2: ] for sr in open(CHAMPION_MODEL + '/labels', 'r').read().splitlines() if
                   len(sr.split('\t')) > 1 and sr.split('\t')[ 1 ] is not 'O' }

        return { document_path: labels }

    labeling = defaultdict()
    for concept, sentences in eval_set.items():
        eval_subset = { concept[1:] : sentences }

        labeling.update( label_document(document=eval_subset, document_path=concept[0]) )

    labeling = { k: filter(lambda x: x in target_keys, v) for k, v in labeling.items() }

    paths = { path[ 0 ] for path in labeling.items() }

    y_pred = [ sorted(labeling[ path ]) for path in paths ]

    paths_labels = { path: open(path, 'r').read().splitlines() for path in paths }
    paths_labels_cleaned = { k: map(str.lower, v) for k, v in paths_labels.items() }
    gold = { k: filter(lambda x: x in target_keys, v) for k, v in paths_labels_cleaned.items() }

    y_true = [ sorted(gold[ path ]) for path in paths ]

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
    labeling_threshold = 4.5,
    window_size = 2,

    subsample = 6,
    target_count = 4,
    min_count = 1000,
    max_count = 10000,
    composed_words = False,
    disjoint = False,
    train_test_evaluate = [ .6, .2, .2 ],

    genetic_generations = 0,
    genetic_tournament_size = 2,
    genetic_selected_from_tournament = 1,
    genetic_champions = 1
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
    :param genetic_generations:
    :param genetic_tournament_size:
    :param genetic_selected_from_tournament:
    :param genetic_champions:
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
        :type genetic_generations: int
        :type genetic_tournament_size: int
        :type genetic_selected_from_tournament: int
        :type genetic_champions: int
    :return: ( embedding_layer, epochs, words_loaded, labeling_threshold, window_size, subsample, target_count,
               min_count, max_count, composed_words, disjoint, train_test_evaluate, genetic_generations,
               genetic_tournament_size, genetic_selected_from_tournament, genetic_champions,
               f1_score macro, f1_score micro, f1_score weighted, accuracy_score normalize=True,
               precision_score micro, precision_score macro, precision_score weighted, recall_score micro,
               recall_score macro, recall_score weighted )
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
    format_input_crf(data=train_set, destination_file=train_file, distance_threshold=labeling_threshold, model=model, window=window_size)
    test_set = load_sentences(test_set)
    format_input_crf(data=test_set, destination_file=test_file, distance_threshold=labeling_threshold, model=model, window=window_size)
    print '--done'

    print '\n--start genetic train'
    genetic_feature_discovery(generations=genetic_generations, tournament_size=genetic_tournament_size, selected_from_tournament=genetic_selected_from_tournament, how_many_champions=genetic_champions)
    print '--done'

    print '--evaluate resulting solution'
    eval_set = load_sentences(eval_set)
    evals = evaluate_solution(eval_set=eval_set, target_keys=target_concepts)
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
    retval += [ str(genetic_generations) ]
    retval += [ str(genetic_tournament_size) ]
    retval += [ str(genetic_selected_from_tournament) ]
    retval += [ str(genetic_champions) ]
    retval += [ str(v) for v in evals ]

    return ', '.join(retval)


os.system('echo "\n" >> %s' % RESULT_FILE_csv)
os.system('echo "\n" >> %s' % RESULT_FILE_csv)
os.system('echo "NEW BULK" >> %s' % RESULT_FILE_csv)

header = [ ]
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
header += [ 'genetic_generations' ]
header += [ 'genetic_tournament_size' ]
header += [ 'genetic_selected_from_tournament' ]
header += [ 'genetic_champions' ]
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
os.system('echo "%s" >> %s' % (', '.join(header) + '\n', RESULT_FILE_csv))

os.system('rm -rf %s' % TEMPLATES)
os.system('rm -rf %s' % MODELS)
os.system('rm -rf %s' % SANDBOX)

os.system('echo "%s" >> %s' % (
        main(embedding_layer='512', epochs='15', words_loaded=50000, labeling_threshold=5, window_size=3, subsample=None,
             target_count=7, min_count=None, max_count=None, composed_words=None, disjoint=None,
             train_test_evaluate=[ .6, .2, .2 ]
        ), RESULT_FILE_csv)
)


print '\n\n--DONE'


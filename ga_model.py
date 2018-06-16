import sys
from time import time
import numpy as np
import pickle
from utils import MLP2
import mnist

log = open('ga_log.txt', 'w')
log.write('generation-best-loss,avg-loss,generation-best-accuracy,avg-accuracy\n')


class EAModel(object):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim, num_elitism):
        # net params
        self.in_dim = in_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.out_dim = out_dim

        # population
        self.pop_size = -1
        self.population = []

        # elitism settings
        self.num_elitism = num_elitism
        self.elitism = []
        self.best = (np.inf, None, -1)

        self.fitness_roulette = []
        self.avg_loss = -1
        self.avg_accuracy = -1
        self.mutate_rate = -1

        # fitness evaluation settings
        self.key_index = -1
        self.reverse = True
        self.cmp = None

    def init_population(self, pop_size):
        """
        Create the initial population and create the roulette wheel.
        :param pop_size: (int) population size.
        """
        # population
        self.pop_size = pop_size
        self.population.extend([MLP2(self.in_dim, self.hid_dim1, self.hid_dim2, self.out_dim)
                                for _ in range(pop_size)])
        # roulette wheel
        for i in range(pop_size - self.num_elitism):
            self.fitness_roulette.extend([i] * (i + 1))

    def calc_fitness(self, dataset_sample):
        """
        evaluate each mlp in the population on data sample and sort the population (from worst to best).
        :param dataset_sample: (array-like of numpy vectors) data sample.
        """
        # calc scores according to some evaluation metric
        scores = []
        for mlp in self.population:
            # acc, loss = mlp.check_on_dataset(dataset_sample)
            acc, loss = mlp.check_on_dataset_batch(dataset_sample)
            scores.append((loss, mlp, acc))
        scores.sort(key=lambda a: a[self.key_index], reverse=self.reverse)
        self.avg_loss = sum([x[0] for x in scores]) / len(scores)
        self.avg_accuracy = sum([x[2] for x in scores]) / len(scores)

        for _ in range(self.num_elitism):  # remove the bad chromosomes
            scores.pop(0)

        # if needed, update best model
        if self.cmp(self.best[self.key_index], scores[-1][self.key_index]):
            self.best = scores[-1]
            # pickle.dump(self.best[1].get_params(), open('ea_params/model_best.params', 'w'))

        # elitism
        self.elitism = scores[-self.num_elitism:]
        self.elitism.sort(key=lambda a: a[self.key_index], reverse=not self.reverse)

        # sort population from worst to best
        self.population = [x[1] for x in scores]

    def select(self):
        """
        select 2 parents from the population according to the roulette wheel.
        :return: 2 MLP2 objects.
        """
        idxs = np.random.choice(self.fitness_roulette, size=2)
        id1, id2 = idxs[0], idxs[1]
        return self.population[id1], self.population[id2]

    def crossover(self, p1, p2):
        """
        take parts of both parents to construct new MLP2.
        :param p1: (MLP2) first parent.
        :param p2: (MLP2) second parent.
        :return: MLP2.
        """
        child_params = []
        for param1, param2 in zip(p1.get_params(), p2.get_params()):
            if len(param1.shape) == 1:  # single vector crossover
                if np.random.random() < 0.5:
                    child_params.append(param1)
                else:
                    child_params.append(param2)
            else:  # matrix crossover according to columns
                param = np.zeros(param1.shape)
                for i in range(param1.shape[1]):
                    if np.random.random() < 0.5:
                        param[:, i] += param1[:, i]
                    else:
                        param[:, i] += param2[:, i]
                child_params.append(param)
        return MLP2(params=child_params)

    def mutate(self, child):
        """
        mutate the given mlp with gaussian noise. In-Place mutation.
        :param child: (MLP2).
        """
        if np.random.random() < self.mutate_rate:
            params = child.get_params()
            val = np.random.random()
            for i in range(len(params)):  # add gaussian noise to each parameter
                if val < 0.3:
                    params[i] += np.random.normal(scale=0.0003, size=params[i].shape)
                elif val < 0.6:
                    params[i] += np.random.normal(scale=0.005, size=params[i].shape)
                else:
                    params[i] += np.random.normal(scale=0.01, size=params[i].shape)
            child.set_params(params)  # update object params

    def run(self, dataset, sample_size, num_generations, mutate_p, mode, best=None):
        """
        run the genetic algorithm.
        :param dataset: (array-like) each is tuple (x, y) where x is numpy array and y is label (int).
        :param sample_size: (int) number of samples to take from dataset to use in each generation.
        :param num_generations: (int) number of generation to run the algorithm.
        :param mutate_p: (float) number from 0 to 1, mutation rate.
        :param mode: (string) evaluation metric, options are [loss, accuracy].
        :param best: tuple of pre-trained MLP2 object to insert the initial population,
                     tuple of form (loss, MLP2, accuracy),
                     or None for no pre-trained mlp.
        """
        if not best:  # no pre-trained
            self.best = (np.inf, None, -1)  # init best chromosome
        else:  # pre-trained
            print 'add pretrained to population'
            self.best = best
            self.population.pop()
            self.population.insert(0, best[1])
        self.mutate_rate = mutate_p

        # update setting of evaluation metric according to mode
        if mode == 'loss':
            self.key_index = 0
            self.reverse = True
            self.cmp = lambda a1, a2: a1 >= a2
        else:
            self.key_index = 2
            self.reverse = False
            self.cmp = lambda a1, a2: a1 <= a2

        data_idx = range(len(dataset))

        for generation in range(num_generations):
            t_start = time()

            # sample dataset and evaluate chromosomes
            np.random.shuffle(data_idx)
            dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
            self.calc_fitness(dataset_sample)
            np.random.shuffle(self.fitness_roulette)  # shuffle roulette

            # new population - add elitism
            curr_pop = [x[1] for x in self.elitism]
            # breed one with best chromosome
            p2 = np.random.choice(self.population)
            child = self.crossover(self.best[1], p2)
            self.mutate(child)
            curr_pop.append(child)

            while len(curr_pop) != self.pop_size:  # new population continious
                p1, p2 = self.select()
                child = self.crossover(p1, p2)
                self.mutate(child)
                curr_pop.append(child)

            # update population
            self.population = curr_pop
            print '{} time: {:.2f} |' \
                  ' best: loss: {:.5f} acc: {:.3f} |' \
                  ' loss: gen-best {:.5f} avg: {:.5f} |' \
                  ' acc: gen-best: {:.3f} avg: {:.3f}'.format(
                generation, time() - t_start, self.best[0], self.best[2],
                self.elitism[0][0], self.avg_loss, self.elitism[0][2], self.avg_accuracy
            )
            log.write('{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
                self.elitism[0][0], self.avg_loss, self.elitism[0][2], self.avg_accuracy
            ))

    def get_best(self, dataset, sample_size):  # always according to accuracy
        """
        evaluate the elitism and the best mlps on dev-set from the given dataset with the given size.
        :param dataset: (array-like) each is tuple of (x, y) where x is numpy array and y is a label (int).
        :param sample_size: (int) size of dev-set.
        :return: the best MLP2 object that performed the best on the dev-set.
        """
        # create dev-set
        data_idx = range(len(dataset))
        np.random.shuffle(data_idx)
        dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]

        # evaluate elitism
        ops = [x[1] for x in self.elitism]
        ls = [(mlp.check_on_dataset(dataset_sample)[0], mlp) for mlp in ops]
        ls.sort(key=lambda a: a[0])
        return ls[-1][1]


def load_dataset(filename, has_labels=True):
    """
    each line in dataset file is in the format: x1 x2 ... xn (separated by space).
    each xi of the input vector is int from 0 to 255.
    if has labels, then the last value in each line is the label (int).
    :param filename: (string) path to the dataset file.
    :param has_labels: (boolean) does the file has labels or not.
    :return: list of object - tuple (has labels) or numpy array (no labels).
    """
    dataset = []

    with open(filename) as f:
        for line in f:
            line = line.split()
            if has_labels:
                label = int(line.pop(-1))
            line = np.array(map(float, line)) / 255  # normalized
            if has_labels:
                line = (line, label)
            dataset.append(line)

    return np.array(dataset)


def test_model(model, dataset_file, output_file):
    """
    craete a prediction file, in each line there is a label.
    :param model: (MLP2) model object.
    :param dataset_file: (string) path to dataset file.
    :param output_file: (string) path to the file tthat will be created.
    """
    dataset = load_dataset(dataset_file, has_labels=False)
    preds = []
    for x in dataset:
        preds.append(model.predict_on(x))

    with open(output_file, 'w') as f:
        f.write('\n'.join(preds))


def eval_model(model, dataset_file):
    """
    check accuracy on dataset and notify the user.
    :param model: (MLP2) model object.
    :param dataset_file: (string) path to dataset file.
    """
    dataset = load_dataset(dataset_file, has_labels=True)
    good = 0.0
    for x, gold in dataset:
        pred = model.predict_on(x)
        if pred == gold:
            good += 1

    print 'accuracy:', good / len(dataset)


def train_on(config_file):
    """
    :param config_file: (string) path to configuration file.
    :return: (MLP2) model.
    """
    # load configurations
    configs = dict()
    with open(config_file) as f:
        for line in f:
            [key, val] = line.split()
            configs[key] = val

    # load data
    print 'loading data'
    mndata = mnist.MNIST(configs[MNIST_PATH])
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x).astype('float32') / 255
    train_data = zip(train_x, train_y)

    # set dims
    in_dim = train_x[0].shape[0]
    hid_dim1 = 128
    hid_dim2 = 64
    out_dim = 10

    # check if to load pre-trained MLP2,
    # set pretrained to be a path to '.params' file if needed to load a pre-trained MLP2.
    best = None
    if PRETRAINED in configs:
        print 'load and eval pretrained on train-data'
        params = pickle.load(open(configs[PRETRAINED]))
        best_mlp = MLP2(params=params)
        acc, loss = best_mlp.check_on_dataset(train_data)
        best = (loss, best_mlp, acc)

    # run GA
    ea_model = EAModel(in_dim, hid_dim1, hid_dim2, out_dim, num_elitism=int(configs[NUM_ELITISM]))
    ea_model.init_population(pop_size=int(configs[POP_SIZE]))  # init step
    ea_model.run(train_data, sample_size=int(configs[SAMPLE_SIZE]), num_generations=int(configs[NUM_GENS]),
                 mutate_p=float(configs[MUTATE_RATE]), mode=configs[METRIC], best=best)
    model = ea_model.get_best(train_data, sample_size=int(configs[DEV_SIZE]))

    # test
    print 'start test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc, test_loss = model.check_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc, 'test-loss:', test_loss

    train_acc, train_loss = model.check_on_dataset(train_data)
    print 'train-acc:', train_acc, 'train-loss:', train_loss
    log.write('\ntrain - accuracy: {:.5f} | loss: {:.5f}'.format(train_acc, train_loss))
    log.write('\ntest - accuracy: {:.5f} | loss: {:.5f}'.format(test_acc, test_loss))

    print 'best acc:', ea_model.best[0]
    return model


TRAIN = '-train'
SAVE = '-save'
LOAD = '-load'
EVAL = '-eval'
TEST = '-test'

PRETRAINED = 'pretrained'
POP_SIZE = 'population_size'
NUM_ELITISM = 'num_elitism'
SAMPLE_SIZE = 'sample_size'
NUM_GENS = 'num_generations'
MUTATE_RATE = 'mutate_rate'
METRIC = 'metric'
DEV_SIZE = 'dev_size'
MNIST_PATH = 'mnist_path'


def main():
    """
    Args - can combine multiple:
        [-train configuration_file]
        [-save model_filename]
        [-load params_file]
        [-eval dataset_file]
        [-test dataset_file output_filename]
    """
    args = sys.argv[1:]

    if TRAIN in args:
        config_file = args[args.index(TRAIN) + 1]
        model = train_on(config_file)

        if SAVE in args:
            target_file = args[args.index(SAVE) + 1]
            pickle.dump(model.get_params(), open(target_file, 'w'))

    if LOAD in args:
        params_file = args[args.index(LOAD) + 1]
        params = pickle.load(open(params_file))
        model = MLP2(params=params)

    if EVAL in args:
        dataset_file = args[args.index(EVAL) + 1]
        eval_model(model, dataset_file)

    if TEST in args:
        i = args.index(TEST)
        dataset_file = args[i + 1]
        output_filename = args[i + 2]
        test_model(model, dataset_file, output_filename)


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0
    log.close()

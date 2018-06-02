from time import time
import numpy as np
import pickle
from utils import MLP2
import mnist


class EAModel(object):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim, num_elitism):
        self.in_dim = in_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.out_dim = out_dim

        self.pop_size = -1
        self.population = []

        self.num_elitism = num_elitism
        self.elitism = []
        self.best = (np.inf, None)

        self.fitness_roulette = []
        self.avg_loss = -1
        self.mutate_rate = -1

    def init_population(self, pop_size):
        self.pop_size = pop_size
        self.population.extend([MLP2(self.in_dim, self.hid_dim1, self.hid_dim2, self.out_dim)
                                for _ in range(pop_size)])
        for i in range(pop_size - self.num_elitism):
            self.fitness_roulette.extend([i] * (i + 1))

    def calc_fitness(self, dataset_sample):
        # calc scores according to loss
        scores = [(mlp.check_on_dataset(dataset_sample)[1], mlp) for mlp in self.population]
        scores.sort(key=lambda a: a[0], reverse=True)
        self.avg_loss = sum([x[0] for x in scores]) / len(scores)

        for _ in range(self.num_elitism):  # remove the bad chromosomes
            scores.pop(0)

        # if needed, update best model
        if self.best[0] >= scores[-1][0]:
            self.best = scores[-1]

        # elitism
        self.elitism = scores[-self.num_elitism:]
        self.elitism.sort(key=lambda a: a[0])

        # sort population from worst to best
        self.population = [x[1] for x in scores]

    def select(self):
        idxs = np.random.choice(self.fitness_roulette, size=2)
        id1, id2 = idxs[0], idxs[1]
        return self.population[id1], self.population[id2]

    def crossover(self, p1, p2):
        child_params = []
        for param1, param2 in zip(p1.get_params(), p2.get_params()):
            if len(param1.shape) == 1:  # single vector crossover
                if np.random.random() < 0.5:
                    child_params.append(param1)
                else:
                    child_params.append(param2)
            else:  # matrix crossover
                param = np.zeros(param1.shape)
                for i in range(param1.shape[1]):
                    if np.random.random() < 0.5:
                        param[:, i] += param1[:, i]
                    else:
                        param[:, i] += param2[:, i]
                child_params.append(param)
        return MLP2(params=child_params)

    def mutate(self, child):
        if np.random.random() < self.mutate_rate:
            params = child.get_params()
            val = np.random.random()
            for i in range(len(params)):  # add gaussian noise to each parameter
                if val < 0.3:
                    params[i] += np.random.normal(scale=0.0001, size=params[i].shape)
                elif val < 0.6:
                    params[i] += np.random.normal(scale=0.001, size=params[i].shape)
                else:
                    params[i] += np.random.normal(scale=0.01, size=params[i].shape)
            child.set_params(params)  # update object params

    def run(self, dataset, population_size, sample_size, num_generations, mutate_p):
        self.mutate_rate = mutate_p
        self.init_population(population_size)  # init step
        data_idx = range(len(dataset))

        for generation in range(num_generations):
            t_start = time()

            # sample dataset and evaluate chromosomes
            np.random.shuffle(data_idx)
            dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
            self.calc_fitness(dataset_sample)
            np.random.shuffle(self.fitness_roulette)  # shuffle roulette

            # new population - add elitism
            curr_pop = [mlp for _, mlp in self.elitism]
            # breed with best one chromosome
            p2 = self.population[0]  # take the current worst chromosome
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
            print '{} time: {:.2f} all-best: {:.5f} gen-best: {:.5f} avg: {:.5f}'.format(
                generation, time() - t_start, self.best[0], self.elitism[0][0], self.avg_loss
            )

    def get_best(self, dataset, sample_size):
        data_idx = range(len(dataset))
        np.random.shuffle(data_idx)
        dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
        ops = [mlp for _, mlp in self.elitism]
        ls = [(mlp.check_on_dataset(dataset_sample)[0], mlp) for mlp in ops]
        ls.sort(key=lambda a: a[0])
        return ls[-1][1]


def main():
    # load data
    print 'loading data'
    mndata = mnist.MNIST('./data')
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x).astype('float32') / 255
    train_data = zip(train_x, train_y)

    # set dims
    in_dim = train_x[0].shape[0]
    hid_dim1 = 128
    hid_dim2 = 64
    out_dim = 10

    ea_model = EAModel(in_dim, hid_dim1, hid_dim2, out_dim, num_elitism=5)
    ea_model.run(train_data, population_size=50, sample_size=100, num_generations=10000, mutate_p=0.4)
    model = ea_model.get_best(train_data, sample_size=12000)

    # blind test
    print 'start blind test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc, test_loss = model.check_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc, 'test-loss:', test_loss

    train_acc, train_loss = model.check_on_dataset(train_data)
    print 'train-acc:', train_acc, 'train-loss:', train_loss
    pickle.dump(model.get_params(),
                open('ea_params/model_{}_{}.params'.format(int(train_acc * 100), int(test_acc * 100)), 'w'))

    print 'best acc:', ea_model.best[0]


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

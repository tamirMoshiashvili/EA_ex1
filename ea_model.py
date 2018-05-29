from time import time
import mnist
from utils import MLP2, np
import pickle


class EA(object):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim, num_elitism=2):
        # save net dims
        self.in_dim = in_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.out_dim = out_dim

        self.population = []
        self.num_population = 0
        self.num_elitism = num_elitism
        self.elitism = [(-1, None)] * num_elitism
        self.fitness_roulette = []

    def init_population(self, size):
        self.population = [MLP2(self.in_dim, self.hid_dim1, self.hid_dim2, self.out_dim) for _ in range(size)]
        for i in range(size - self.num_elitism):
            self.fitness_roulette.extend([i] * (i + 1))
        self.num_population = size

    def calc_fitness(self, dataset_sample):
        # calc accuracy of each mlp2 on the given dataset-sample
        scores = [(mlp.check_on_dataset(dataset_sample)[0], mlp) for mlp in self.population]
        scores.sort(key=lambda a: a[0])
        self.population = [x[1] for x in scores]
        for _ in range(self.num_elitism):   # remove the bad chromosomes
            self.population.pop(0)

        self.elitism = scores[-self.num_elitism:]
        self.elitism.reverse()
        np.random.shuffle(self.fitness_roulette)  # shuffle roulette

    def select_parents(self):
        id1 = np.random.choice(self.fitness_roulette)
        id2 = np.random.choice(self.fitness_roulette)
        while id1 == id2:
            id2 = np.random.choice(self.fitness_roulette)
        return self.population[id1], self.population[id2]

    def crossover(self, p1, p2):
        child_params1, child_params2 = [], []

        for param1, param2 in zip(p1.get_params(), p2.get_params()):
            if len(param1.shape) == 1:  # single vector crossover
                cut_point = np.random.randint(param1.shape[0])
                child_params1.append(np.concatenate((param1[0:cut_point], param2[cut_point:])))
                child_params2.append(np.concatenate((param2[0:cut_point], param1[cut_point:])))
            else:  # matrix crossover
                shape = param1.shape
                mat1, mat2 = np.zeros(shape), np.zeros(shape)
                for i in range(shape[0]):  # pick randomly rows from each matrix
                    if np.random.random() < 0.5:
                        mat1[i] += param1[i]
                        mat2[i] += param2[i]
                    else:
                        mat1 += param2[i]
                        mat2 += param1[i]

                child_params1.append(mat1)
                child_params2.append(mat2)

        return MLP2(params=child_params1), MLP2(params=child_params2)

    def mutate(self, child, p=0.05):
        if np.random.random() < p:
            # add gaussian noise to each parameter
            params = child.get_params()
            # i = np.random.choice(range(len(params)))
            # params[i] += np.random.normal(scale=0.001, size=params[i].shape)
            for i in range(len(params)):
                params[i] += np.random.normal(scale=0.001, size=params[i].shape)
            # update object params
            child.set_params(params)

    def run(self, dataset, population_size=20, sample_size=200, num_generations=1000):
        self.init_population(size=population_size)  # init step
        data_idx = range(len(dataset))

        for generation in range(num_generations):
            t_start = time()

            # sample dataset and evaluate chromosomes
            dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
            self.calc_fitness(dataset_sample)

            # initiate new population, starting with elitism
            curr_population = [mlp for _, mlp in self.elitism]

            # create new population
            while len(curr_population) != self.num_population:
                p1, p2 = self.select_parents()
                child1, child2 = self.crossover(p1, p2)
                self.mutate(child1)
                self.mutate(child2)
                curr_population.append(child1)
                curr_population.append(child2)

            # update population
            self.population = curr_population
            print generation, 'time:', time() - t_start,\
                'best1:', self.elitism[0][0], 'best2:', self.elitism[1][0]  # TODO

    def get_best(self):
        return self.elitism[0][1]


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

    ea_model = EA(in_dim, hid_dim1, hid_dim2, out_dim)
    ea_model.run(train_data)
    model = ea_model.get_best()

    # blind test
    print 'start blind test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc, test_loss = model.check_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc, 'test-loss:', test_loss

    train_acc, train_loss = model.check_on_dataset(train_data)
    print 'train-acc:', train_acc, 'train-loss:', train_loss
    pickle.dump(model.get_params(), open('model_{}_{}.params'.format(int(train_acc * 100), int(test_acc * 100)), 'w'))


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

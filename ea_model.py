from time import time
import mnist
from utils import MLP2, np, Glorot_init
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
        self.gen_elitism = [(-1, None)] * num_elitism
        self.abs_elitism = [(-1, None)] * num_elitism
        self.fitness_roulette = []

        self.best = (-1, None)
        self.avg_acc = -1

    def init_population(self, size):
        self.population = [MLP2(self.in_dim, self.hid_dim1, self.hid_dim2, self.out_dim) for _ in range(size)]
        for i in range(size):
            self.fitness_roulette.extend([i] * (i + 1))
        self.num_population = size

    def calc_fitness(self, dataset_sample):
        # calc accuracy of each mlp2 on the given dataset-sample
        scores = [(mlp.check_on_dataset(dataset_sample)[0], mlp) for mlp in self.population]
        scores.sort(key=lambda a: a[0])
        self.avg_acc = sum([x[0] for x in scores]) / len(scores)
        for _ in range(self.num_elitism):  # remove the bad chromosomes
            self.population.pop(0)

        self.gen_elitism = scores[-self.num_elitism:]
        self.gen_elitism.reverse()
        if self.best[0] <= self.gen_elitism[0][0]:
            self.best = self.gen_elitism[0]

        self.abs_elitism.extend(self.gen_elitism)
        self.abs_elitism.sort(key=lambda a: a[0], reverse=True)
        self.abs_elitism = self.abs_elitism[:self.num_elitism]

        scores.extend(self.abs_elitism)
        self.population = [x[1] for x in scores]

        for i, (_, mlp) in enumerate(self.abs_elitism):
            self.abs_elitism[i] = ((mlp.check_on_dataset(dataset_sample)[0] + self.abs_elitism[i][0]) / 2, mlp)

    def select_parents(self):
        id1 = np.random.choice(self.fitness_roulette)
        id2 = np.random.choice(self.fitness_roulette)
        while id1 == id2:
            id2 = np.random.choice(self.fitness_roulette)
        return self.population[id1], self.population[id2]

    def crossover(self, p1, p2):
        child_params = []
        for param1, param2 in zip(p1.get_params(), p2.get_params()):
            if np.random.random() < 0.5:
                child_params.append(param1 + param2 / 10)
            else:
                child_params.append(param1 / 10 - param2)
        return MLP2(params=child_params)
        # child_params = []
        #
        # for param1, param2 in zip(p1.get_params(), p2.get_params()):
        #     if len(param1.shape) == 1:  # single vector crossover
        #         if np.random.random() < 0.5:
        #             child_params.append(param1)
        #         else:
        #             child_params.append(param2)
        #     else:  # matrix crossover
        #         shape = param1.shape
        #         mat1 = np.zeros(shape)
        #         for i in range(shape[0]):  # pick randomly rows from each matrix
        #             if np.random.random() < 0.5:
        #                 mat1[i] += param1[i]
        #             else:
        #                 mat1 += param2[i]
        #
        #         child_params.append(mat1)
        #
        # return MLP2(params=child_params)

    def mutate(self, child, p=0.5):
        if np.random.random() < p:
            # add gaussian noise to each parameter
            params = child.get_params()
            for i in range(len(params)):
                if np.random.random() < p:
                    params[i] += np.random.normal(scale=0.01, size=params[i].shape)
                else:
                    params[i] += np.random.normal(scale=0.1, size=params[i].shape)
            # update object params
            child.set_params(params)

    def run(self, dataset, population_size=20, sample_size=200, num_generations=1000, mutate_p=0.5):
        self.init_population(size=population_size)  # init step
        data_idx = range(len(dataset))

        for generation in range(num_generations):
            t_start = time()

            # sample dataset and evaluate chromosomes
            np.random.shuffle(data_idx)
            dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
            self.calc_fitness(dataset_sample)
            np.random.shuffle(self.fitness_roulette)  # shuffle roulette

            # initiate new population, starting with elitism
            curr_population = [mlp for _, mlp in self.gen_elitism + self.abs_elitism]
            p1 = self.best[1]   # todo add choice from best and gen-elit and abs-elit
            for _ in range(self.num_elitism):  # breed the best mlp
                p2 = np.random.choice(self.population)
                child = self.crossover(p1, p2)
                self.mutate(child, mutate_p)
                curr_population.append(child)

            # create new population
            while len(curr_population) != self.num_population:
                p1, p2 = self.select_parents()
                child = self.crossover(p1, p2)
                self.mutate(child, mutate_p)
                curr_population.append(child)

            # update population
            self.population = curr_population
            print generation, 'time:', time() - t_start, 'all-best:', self.best[0], \
                'gen-best:', self.gen_elitism[0][0], 'avg-gen-acc:', self.avg_acc

    def get_best(self, dataset, sample_size=1000):
        data_idx = range(len(dataset))
        np.random.shuffle(data_idx)
        dataset_sample = [dataset[i] for i in np.random.choice(data_idx, sample_size)]
        ops = [mlp for _, mlp in self.gen_elitism + self.abs_elitism]
        ops.append(self.best[1])
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

    ea_model = EA(in_dim, hid_dim1, hid_dim2, out_dim, num_elitism=5)
    ea_model.run(train_data, population_size=30, sample_size=100, num_generations=10000, mutate_p=0.5)
    model = ea_model.get_best(train_data, sample_size=12000)

    # blind test
    print 'start blind test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc, test_loss = model.check_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc, 'test-loss:', test_loss

    train_acc, train_loss = model.check_on_dataset(train_data)
    print 'train-acc:', train_acc, 'train-loss:', train_loss
    pickle.dump(model.get_params(), open('ea_params/model_{}_{}.params'.format(int(train_acc * 100), int(test_acc * 100)), 'w'))

    print 'best acc:', ea_model.best[0]


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

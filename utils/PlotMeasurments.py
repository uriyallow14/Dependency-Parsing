import matplotlib.pyplot as plt


def plot_measurement_graph(pair_of_measure_list, stages, measure_name, version):
    plt.figure()
    epochs_num = list(range(len(pair_of_measure_list[0])))
    plt.plot(epochs_num, pair_of_measure_list[0], label=stages[0])
    plt.plot(epochs_num, pair_of_measure_list[1], label=stages[1])
    plt.title(f'{measure_name}')
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel(measure_name)
    plt.savefig(f'results\\{measure_name}_{version}')

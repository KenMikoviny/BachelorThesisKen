""" Script that is used to generate plots from result dictionaries obtained from training the models"""

from utility_methods import load_obj
import pprint
import matplotlib.pyplot as plt

# For visualization.
import matplotlib.pyplot as plt
from utility_methods import initialize_embeddings

training_results = load_obj("training_results")
transfered_entity_results = load_obj("transfered_entity_results_dict")
combined_model_training_results = load_obj("combined_model_training_results")
epoch_list = training_results.keys()

print(training_results[0])

def generate_figure(epochs, type_training_epochs, name_on_figure, dict_key_name, xlim = None, ylim = None):
    """
    Generates and saves figure chosen by params in mpqe/figures/
    to see a list of dict_key_name do for ex. print(training_results[0])

    :param type_training_epochs: embeddings to visualize
    :param name_on_figure: entity labels for coloring
    :param dict_key_name: statistic to plot
    :param xlim: x axis limit
    :param ylim: y axis limit
    """ 
    data_entity_model, data_type_model = [], []

    for key in training_results:
        data_entity_model.append(training_results[key][dict_key_name])

    for key in combined_model_training_results:
        data_type_model.append(combined_model_training_results[key][dict_key_name])


    fig = plt.figure(tight_layout=True)

    plt.plot(epoch_list, data_entity_model, label = "Entity MPQE Model", color='red')
    plt.plot(epoch_list, data_type_model, label = "Combined MPQE Model", color='green')

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel('Epochs')
    plt.ylabel(dict_key_name)
    plt.title(name_on_figure + '\n (' + str(type_training_epochs) + ' epochs of training before weight transfer)') 
    plt.legend()
    plt.show()
    fig.savefig('mpqe/figures/' + dict_key_name + '_' + str(type_training_epochs) + 'e' + '.png')


# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Average hits at 3",
#     dict_key_name="avg.hits_at_3",
# )

# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Worst hits at 3",
#     dict_key_name="worst.hits_at_3",
# )

# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Average hits at 10",
#     dict_key_name="avg.hits_at_10",
# )

# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Worst hits at 10",
#     dict_key_name="worst.hits_at_10",
# )

# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Average mean rank",
#     dict_key_name="avg.mean_rank",
#     xlim=[0,600],
#     ylim=None,
# )

# generate_figure(
#     epochs=1000,
#     type_training_epochs=100,
#     name_on_figure="Average mean rank",
#     dict_key_name="avg.mean_rank",
#     xlim=[0,600],
#     ylim=None,
# )




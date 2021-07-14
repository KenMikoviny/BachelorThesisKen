""" Script that is used to generate plots from result dictionaries obtained from training the models"""

import config
from utility_methods import load_obj
import pprint
import matplotlib.pyplot as plt
from utility_methods import initialize_embeddings

# For embedding visualization
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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


def apply_pca_and_tsne(embeddings, target_dimensions, seed = None):
    """ Apply pca and tsne dimensionality reduction """ 

    # Set n_components to target dimension or min(number of entities, embedding dimension)
    components = min(target_dimensions, min(embeddings.shape[0], embeddings.shape[1]))
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(embeddings.clone().detach().numpy())
    print(pca_result.shape)
    print('Cumulative explained variation for ' + str(components) + ' principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

    time_start = time.time()

    #remove metric cosine to use euclidean distance instead
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000, random_state=seed)
    tsne_pca_results = tsne.fit_transform(embeddings.clone().detach().numpy())
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tsne_pca_results


def save_visualization_of_embeddings(embeddings, labels, target_dimensions, file_name, seed = None):
    """
    Visualization referrence: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    Generates and saves a visualisation of entity embeddings using pca+tsne
    Generates 2 copies in figures folder, first one with labels and axes and second one without

    :param embeddings: embeddings to visualize
    :param labels: entity labels for coloring
    :param target_dimensions: target dimension of pca reduction
    :param file_name: name of the file
    :param seed: seed for reproducible results when using TSNE
    """ 
    
    tsne_pca_results = apply_pca_and_tsne(embeddings,target_dimensions, seed)

    fig = plt.figure(figsize=(16,10))

    unique_labels = len(np.unique(labels))
    sns.scatterplot(
        x=tsne_pca_results[:,0], y=tsne_pca_results[:,1],
        hue=labels,
        palette=sns.color_palette("hls", unique_labels),
        legend="full",
        alpha=0.9,
    )
    fig.savefig(str(config.data_root) + '/figures/' + str(file_name) + '.png')

    fig_without_labels = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_pca_results[:,0], y=tsne_pca_results[:,1],
        hue=labels,
        palette=sns.color_palette("hls", unique_labels),
        legend=False,
        alpha=0.9,
    )
    fig_without_labels.gca().axes.xaxis.set_visible(False)
    fig_without_labels.gca().axes.yaxis.set_visible(False)
    fig_without_labels.gca().grid(True)
    fig_without_labels.savefig(str(config.data_root) + '/figures/' + str(file_name) + '_nolegend' + '.png')

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


combined_before = load_obj("combined_embeddings_before")
combined_after = load_obj("combined_embeddings_after")
entity_before = load_obj("entity_embeddings_before")
entity_after = load_obj("entity_embeddings_after")
type_before = load_obj("type_embeddings_before")
type_after = load_obj("type_embeddings_after")

save_visualization_of_embeddings(combined_before['node_embeddings'], combined_before['labels'], 80, 'combined_embeddings_before', config.tsne_seed)
save_visualization_of_embeddings(combined_after['node_embeddings'], combined_after['labels'], 80, 'combined_embeddings_after', config.tsne_seed)

save_visualization_of_embeddings(entity_before['node_embeddings'], entity_before['labels'], 80, 'entity_embeddings_before', config.tsne_seed)
save_visualization_of_embeddings(entity_after['node_embeddings'], entity_after['labels'], 80, 'entity_embeddings_after', config.tsne_seed)

save_visualization_of_embeddings(type_before['node_embeddings'], type_before['labels'], 80, 'type_embeddings_before', config.tsne_seed)
save_visualization_of_embeddings(type_after['node_embeddings'], type_after['labels'], 80, 'type_embeddings_after', config.tsne_seed)



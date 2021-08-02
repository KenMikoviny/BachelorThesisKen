""" Script that is used to generate plots from result dictionaries obtained from training the models"""
import config
from utility_methods import load_obj
import matplotlib.pyplot as plt
from utility_methods import initialize_embeddings

# For embedding visualization
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from statistics import mean



# Load data from 3 experiments for adding stdev
training_results = load_obj("entity_model_results_1000")
training_results2 = load_obj("entity_model_results_1000_2")
training_results3 = load_obj("entity_model_results_1000_3")
combined_model_training_results = load_obj("combined_model_training_results_1000")
combined_model_training_results2 = load_obj("combined_model_training_results_1000_2")
combined_model_training_results3 = load_obj("combined_model_training_results_1000_3")
transfered_entity_results = load_obj("transfered_entity_results_dict")
epoch_list = training_results.keys()


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
    mean_values_type, stdev_values_type, mean_values_entity, stdev_values_entity = [], [], [], []

    for key in training_results:
        mean_values_entity.append(mean([training_results[key][dict_key_name], training_results2[key][dict_key_name], training_results3[key][dict_key_name]]))
        stdev_values_entity.append(np.std([training_results[key][dict_key_name], training_results2[key][dict_key_name], training_results3[key][dict_key_name]]))
    for key in combined_model_training_results:
        mean_values_type.append(mean([combined_model_training_results[key][dict_key_name], combined_model_training_results2[key][dict_key_name], combined_model_training_results3[key][dict_key_name]]))
        stdev_values_type.append(np.std([combined_model_training_results[key][dict_key_name], combined_model_training_results2[key][dict_key_name], combined_model_training_results3[key][dict_key_name]]))


    fig = plt.figure(tight_layout=True,figsize=(8,8))

    plt.plot(epoch_list, mean_values_entity, label = "Entity MPQE Model", color='red')
    plt.plot(epoch_list, mean_values_type, label = "Combined MPQE Model", color='green')
    #Adding stdev
    plt.fill_between(epoch_list,list(np.array(mean_values_entity)-np.array(stdev_values_entity)),list(np.array(mean_values_entity)+np.array(stdev_values_entity)),alpha=1)
    plt.fill_between(epoch_list,list(np.array(mean_values_type)-np.array(stdev_values_type)),list(np.array(mean_values_type)+np.array(stdev_values_type)),alpha=1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.text(670, -0.4, 'Entity MPQE ' + r'$\sigma$' + ' = ' + str(np.std(data_entity_model))[0:5] + '\n' + 'Combined MPQE ' + r'$\sigma$' + ' = ' + str(np.std(data_type_model))[0:5], fontsize=10)
    plt.xlabel('Epochs')
    plt.ylabel(dict_key_name)
    plt.title(name_on_figure + '\n (' + str(type_training_epochs) + ' epochs of training on types before weight transfer)') 
    plt.legend()
    plt.show()
    fig.savefig('mpqe/figures/' + dict_key_name + '_' + str(type_training_epochs) + 'e' + '.png')


def apply_pca_and_tsne(embeddings, target_dimensions, seed = None):
    """ Apply pca and tsne dimensionality reduction """ 

    # Set n_components to target dimension or min(number of entities, embedding dimension)
    components = min(target_dimensions, min(embeddings.shape[0], embeddings.shape[1]))

    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(embeddings.clone().detach().numpy())
    print('Cumulative explained variation for ' + str(components) + ' principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

    time_start = time.time()

    #remove metric cosine to use euclidean distance instead
    tsne = TSNE(n_components=2, verbose=1000, perplexity=40, n_iter=1000, random_state=seed)
    tsne_pca_results = tsne.fit_transform(pca_result)
    
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
    
    type_after = load_obj("type_embeddings_before")
    unique_labels = len(np.unique(labels))
        
    tsne_pca_results = apply_pca_and_tsne(embeddings,target_dimensions, seed)

    fig = plt.figure(figsize=(10,10))
    
    sns.scatterplot(
        x=tsne_pca_results[:,0], y=tsne_pca_results[:,1],
        hue=labels,
        palette=sns.color_palette("hls", unique_labels),
        legend="full",
        alpha=0.9,
    )



    fig.savefig(str(config.data_root) + '/figures/' + str(file_name) + '.png')

    fig_without_labels = plt.figure(figsize=(10,10))
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

def generate_figures():
    generate_figure(
        epochs=1000,
        type_training_epochs=100,
        name_on_figure="Average hits at 3 over 1000 epochs of training on entities",
        dict_key_name="avg.hits_at_3",
    )

    generate_figure(
        epochs=1000,
        type_training_epochs=100,
        name_on_figure="Worst hits at 3 over 1000 epochs of training on entities",
        dict_key_name="worst.hits_at_3",
    )

    generate_figure(
        epochs=1000,
        type_training_epochs=100,
        name_on_figure="Average hits at 10 over 1000 epochs of training on entities",
        dict_key_name="avg.hits_at_10",
    )

    generate_figure(
        epochs=1000,
        type_training_epochs=100,
        name_on_figure="Worst hits at 10 over 1000 epochs of training on entities",
        dict_key_name="worst.hits_at_10",
    )

    generate_figure(
        epochs=1000,
        type_training_epochs=100,
        name_on_figure="Average mean rank over 1000 epochs of training on entities",
        dict_key_name="avg.mean_rank",
        xlim=[0,600],
        ylim=None,
    )

generate_figures()
# combined_before = load_obj("combined_embeddings_before")
# combined_after = load_obj("combined_embeddings_after")
# entity_before = load_obj("entity_embeddings_before")
# entity_after = load_obj("entity_embeddings_after")
# type_before = load_obj("type_embeddings_before")
# type_after = load_obj("type_embeddings_after")

# save_visualization_of_embeddings(entity_before['node_embeddings'], entity_before['labels'], 100, 'entity_embeddings_before', config.tsne_seed)
# save_visualization_of_embeddings(entity_after['node_embeddings'], entity_after['labels'], 100, 'entity_embeddings_after', config.tsne_seed)

# save_visualization_of_embeddings(type_before['node_embeddings'], type_before['labels'], 80, 'type_embeddings_before', config.tsne_seed)
# save_visualization_of_embeddings(type_after['node_embeddings'], type_after['labels'], 80, 'type_embeddings_after', config.tsne_seed)



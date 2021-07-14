import pathlib


root = pathlib.Path(__file__, "..", "..").resolve()
mpqe_root = root.joinpath("mpqe")
data_root = mpqe_root.joinpath("data")
binary_queries_root = data_root.joinpath("binary_queries")
saved_data_root = data_root.joinpath("saved_data")

aifb_entity_id_path = data_root.joinpath("triple_split_AIFB", "entity_id_typing.txt")
am_entity_id_path = data_root.joinpath("triple_split_AM", "entity_id_typing.txt")
mutag_entity_id_path = data_root.joinpath("triple_split_MUTAG", "entity_id_typing.txt")

binary_queries_aifb_path = binary_queries_root.joinpath("binary_queries_AIFB")

get_new_dataloaders = True
save_embeddings = True
save_results = True

tsne_seed = 10
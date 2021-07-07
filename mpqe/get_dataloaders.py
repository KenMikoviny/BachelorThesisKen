import pathlib

from mphrqe.data import loader
from utility_methods import save_obj

#NOTE: when changing dataset also change entoid and reltoid in mphrqe/data/mappings, this is due to how mapping.py retrieves highest entity index
# TODO add all other query types
# train_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), 
#                 loader.resolve_sample("/2i/*:atmost1000"), loader.resolve_sample("/2i-1hop/*:atmost1000"),
#                 loader.resolve_sample("/3hop/*:atmost1000"), loader.resolve_sample("/3i/*:atmost1000"),
#                 ]
# valid_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), loader.resolve_sample("/3hop/*:atmost1000"), 
#                 loader.resolve_sample("/3i/*:atmost1000"),
#                 ]
# test_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), loader.resolve_sample("/3hop/*:atmost1000"), 
#                loader.resolve_sample("/3i/*:atmost1000"),
#                ]

train_samples = [loader.resolve_sample("/1hop/*:atmost500"), loader.resolve_sample("/1hop-2i/*:atmost500"),
                loader.resolve_sample("/3hop/*:atmost500"),
                ]
valid_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"),
                loader.resolve_sample("/3hop/*:atmost1000"),
                ]
test_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"),
                loader.resolve_sample("/3hop/*:atmost1000"),
                ]

root = pathlib.Path(__file__, "..", "..").resolve()
data_root = root.joinpath("data")
binary_queries_root = data_root.joinpath("binary_queries")
binary_queries_aifb_root = binary_queries_root.joinpath("binary_queries_AIFB")
print(root)
# Loading binary queries directly as data loaders
# dataloaders = loader.get_query_data_loaders(batch_size=16, 
#                                             num_workers=2, 
#                                             data_root=pathlib.Path("/mnt/c/Users/Sorys/OneDrive/VU am/Thesis/BachelorThesisKen/mpqe/data/binary_queries/binary_queries_AIFB"),
#                                             train=train_samples, 
#                                             validation=valid_samples, 
#                                             test=test_samples)

# save_obj(dataloaders, "dataloaders")
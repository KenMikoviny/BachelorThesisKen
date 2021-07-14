import config
import sys
sys.path.append("..")

from BachelorThesisKen.hqe.src.mphrqe.data import loader
from utility_methods import save_obj

def get_dataloaders():
    """ Generate samples and load them into dataloaders """
    #NOTE: when changing dataset also change entoid and reltoid in mphrqe/data/mappings, this is due to how mapping.py retrieves highest entity index
    train_samples = [loader.resolve_sample("/1hop/0qual:atmost1000"), loader.resolve_sample("/1hop-2i/0qual:atmost1000"), loader.resolve_sample("/2hop/0qual:atmost1000"), 
                    loader.resolve_sample("/2i/0qual:atmost1000"), loader.resolve_sample("/2i-1hop/0qual:atmost1000"),
                    loader.resolve_sample("/3hop/0qual:atmost1000"), loader.resolve_sample("/3i/0qual:atmost1000"),
                    ]
    valid_samples = [loader.resolve_sample("/1hop/0qual:atmost1000"), loader.resolve_sample("/1hop-2i/0qual:atmost1000"), loader.resolve_sample("/2hop/0qual:atmost1000"), 
                    loader.resolve_sample("/2i/0qual:atmost1000"), loader.resolve_sample("/2i-1hop/0qual:atmost1000"),
                    loader.resolve_sample("/3hop/0qual:atmost1000"), loader.resolve_sample("/3i/0qual:atmost1000"),
                    ]
    test_samples = [loader.resolve_sample("/1hop/0qual:atmost1000"), loader.resolve_sample("/1hop-2i/0qual:atmost1000"), loader.resolve_sample("/2hop/0qual:atmost1000"), 
                    loader.resolve_sample("/2i/0qual:atmost1000"), loader.resolve_sample("/2i-1hop/0qual:atmost1000"),
                    loader.resolve_sample("/3hop/0qual:atmost1000"), loader.resolve_sample("/3i/0qual:atmost1000"),
                    ]

    # # Simplified dataset for testing
    # train_samples = [loader.resolve_sample("/1hop/*:atmost500"), loader.resolve_sample("/1hop-2i/*:atmost500"),
    #                 loader.resolve_sample("/3hop/*:atmost500"),
    #                 ]
    # valid_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"),
    #                 loader.resolve_sample("/3hop/*:atmost1000"),
    #                 ]
    # test_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"),
    #                 loader.resolve_sample("/3hop/*:atmost1000"),
    #                 ]
    #Loading binary queries directly as data loaders
    dataloaders = loader.get_query_data_loaders(batch_size=32, 
                                                num_workers=2, 
                                                data_root=config.binary_queries_aifb_path,
                                                train=train_samples, 
                                                validation=valid_samples, 
                                                test=test_samples)

    save_obj(dataloaders, "dataloaders")
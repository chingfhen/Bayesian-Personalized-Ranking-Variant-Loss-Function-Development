import argparse


"""
Parses the Model Hyperparameters and Dataset
"""
def parse_args(model):
    
    parse = argparse.ArgumentParser()
    
    # model hyperparameters
    if model == "LightGCN":
        parse.add_argument('--embedding_size', type=int, required=True,
                                           help="size of user/item embeddings")
        parse.add_argument('--n_layers', type=int, required=True,
                                           help="number of layers")
        parse.add_argument('--reg_weight', type=float, required=True, 
                                           help="amount of regularization on model weights")
        parse.add_argument('--require_pow', type=bool, required=True,  
                                           help="whether to square the weights when computing embedding loss")
    elif model == "SimpleX":
        parse.add_argument('--embedding_size', type=int, required=True,
                                           help="size of user/item embeddings")
        parse.add_argument('--margin', type=float, required=True,
                                           help="threshold for hard negative discrimination")
        parse.add_argument('--negative_weight', type=int, required=True,
                                           help="balancing pos and neg losses in CCL")
        parse.add_argument('--gamma', type=float, required=True,
                                           help="balance between user embeddings and positive item embeddings")
        parse.add_argument('--neg_seq_len', type=int, required=True,
                                           help="number of negative samples per positive item")
        parse.add_argument('--reg_weight', type=float, required=True,
                                           help="amount of regularization on model weights")
        parse.add_argument('--aggregator', type=str, required=True, choices = ["mean", "user_attention", "self_attention"],
                                           help="method of aggregating positive item embeddings")
        parse.add_argument('--history_len', type=int, required=True,
                                           help="number of positive item embeddings to aggregate to user")
        parse.add_argument('--require_pow', type=bool, required=True,
                                           help="whether to square the weights when computing embedding loss")
        
    # other important configs
    parse.add_argument('--dataset', type=str, choices = ["ml-100k","gowalla"], required=True,
                       help="Name of dataset")
    
    return parse.parse_args()

if __name__ == "__main__":
    
    args = parse_args("LightGCN")
    print(args)
    
    
    
    
    
    
    
    
    
    
    
# def parse_args():
    
#     # firse parse - parse model
#     parse = argparse.ArgumentParser(description="Choose model")
#     parse.add_argument('--model', type=str, choices = ["lightgcn", "simplex"], required=True,
#                        help="Name of model")
#     args, unknown = parse.parse_known_args()
    
#     # second parse
#     parse = argparse.ArgumentParser(description="Choose hyperparameters")
    
#     # parse model hyperparameters
#     if args.model.lower() == "lightgcn":
#         parse.add_argument('--embedding_size', type=int, required=True,
#                                            help="size of user/item embeddings")
#         parse.add_argument('--n_layers', type=int, required=True,
#                                            help="number of layers")
#         parse.add_argument('--reg_weight', type=float, required=True, 
#                                            help="amount of regularization on model weights")
#         parse.add_argument('--require_pow', type=bool, required=True,  
#                                            help="whether to square the weights when computing embedding loss")
#     elif args.model.lower() == "simplex":
#         parse.add_argument('--embedding_size', type=int, required=True,
#                                            help="size of user/item embeddings")
#         parse.add_argument('--margin', type=float, required=True,
#                                            help="threshold for hard negative discrimination")
#         parse.add_argument('--negative_weight', type=int, required=True,
#                                            help="balancing pos and neg losses in CCL")
#         parse.add_argument('--gamma', type=float, required=True,
#                                            help="balance between user embeddings and positive item embeddings")
#         parse.add_argument('--neg_seq_len', type=int, required=True,
#                                            help="number of negative samples per positive item")
#         parse.add_argument('--reg_weight', type=float, required=True,
#                                            help="amount of regularization on model weights")
#         parse.add_argument('--aggregator', type=str, required=True, choices = ["mean", "user_attention", "self_attention"],
#                                            help="method of aggregating positive item embeddings")
#         parse.add_argument('--history_len', type=int, required=True,
#                                            help="number of positive item embeddings to aggregate to user")
#         parse.add_argument('--require_pow', type=bool, required=True,
#                                            help="whether to square the weights when computing embedding loss")
    
#     # parse all other configurations
#     parse.add_argument('--dataset', type=str, choices = ["ml-100k"], required=True,
#                                        help="Name of dataset")
#     parse.add_argument('--epochs', type=int, required=True,
#                         help="number of training epochs")
#     parse.add_argument('--eval_step', type=int, required=True,
#                         help="The number of training epochs before an evaluation on the valid dataset. \
#                         If it is less than 1, the model will not be evaluated on the valid dataset.")
# #     parse.add_argument('--config_file_list', type=list, default = ['test0.yaml'],
# #                         help="Name of dataset")
#     parse.add_argument('--show_progress', type=bool, default = False,
#                         help="Whether or not to show the progress bar of training and evaluation epochs")
#     parse.add_argument('--log_wandb', type=bool, default = False,
#                         help="Whether to log results to Weights n Biases")
# #     parse.add_argument('--eval_args', type=dict, default = {"split": {"RS": [6,2,2]}},
# #                         help="Split ratio")
# #     parse.add_argument('--metrics', type=list, default = ["Recall","NDCG"],
# #                         help="Evaluation metrics")
#     parse.add_argument('--valid_metric', type=str, default = "NDCG@20",
#                         help="Metric for early stopping")
# #     parse.add_argument('--topk', type=list, default = [10,20],
# #                         help="Value of k for topk evaluation metrics")
    
    
#     args,unknown = parse.parse_known_args()
    
#     return args, unknown
    

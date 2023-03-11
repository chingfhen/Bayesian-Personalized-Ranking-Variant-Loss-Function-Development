"""
This script trains and evaluates SimpleX, then saves the performance
"""
from parse import parse_args
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
from logging import getLogger

def define_run_name(model, config_dict):
    embedding_size = config_dict["embedding_size"]  
    margin = config_dict["margin"]  
    negative_weight = config_dict["negative_weight"]  
    gamma = config_dict["gamma"] 
    history_len = config_dict["history_len"] 
    reg_weight = config_dict["reg_weight"]  
    neg_seq_len = config_dict["neg_seq_len"]  
    return f"{model}-{embedding_size}-{margin}-{gamma}-{neg_seq_len}-{negative_weight}-{history_len}-{reg_weight}"
def define_run_note(model, config_dict):
    run_name_definition = "{model}-{embedding_size}-{margin}-{gamma}-{neg_seq_len}-{negative_weight}-{history_len}-{reg_weight}"
    return f"{run_name_definition}\n{str(config_dict)}"


if __name__ == "__main__":
    
    model = "SimpleX"
    evaluate_on_test = False
  
    # get config
    config_dict = vars(parse_args(model))
    config_dict["name"] = define_run_name(model, config_dict)
    config_dict["notes"] = define_run_note(model, config_dict)
    config_dict["neg_sampling"] = {"uniform": config_dict["neg_seq_len"]}
    config = Config(model=model, dataset=config_dict["dataset"], config_file_list=["default_config.yaml", "main_config.yaml"],
                    config_dict=config_dict)
    print(config)
    
    # get logger
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    
    # get and prep dataset
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # get model
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # get trainer 
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # train model 
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=evaluate_on_test, show_progress=config['show_progress']
    )

    # evaluate model 
    
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    test_result = None
    if evaluate_on_test:
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    

    results = {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
    print(results)





    


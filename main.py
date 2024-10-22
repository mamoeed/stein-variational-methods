"""
main file to run everything based on config parameters

"""
import sys
import yaml
from experiment import run_experiment
import numpy as np


def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = read_config(config_path)

    print("Configuration loaded successfully:")

    # run multiple experiments of different seed values and save results
    seeds = [38,39,40,41,42] #  [38,39,40,41,42]
    num_seeds = len(seeds)
    test_results = []
    for s in seeds:
        print(f'\n############ Starting new experiment, seed: {s} #############\n')
        test_results.append(run_experiment(config, s))
    print('### CONFIG: ###')
    print('learning rate:',config['lr'])
    print('method:',config['method'])
    print('anneal:',config['anneal'])

    print("\n\nRESULTS OVER ALL SEED RUNS:")
    
    if config['is_classification']:
        test_NLL, test_accuracy, test_cross_entropy, test_brier, test_entropy, test_auroc = [],[],[],[],[],[]
        for r in test_results:
            print(r)
            test_accuracy.append(r['Accuracy'])
            test_NLL.append(r['NLL'])
            test_cross_entropy.append(r['Cross Entropy'])
            test_brier.append(r['Brier'])
            test_entropy.append(r['Entropy'])
            test_auroc.append(r['AUROC'])
        print('VALIDATION NLL...\n')
        [print(r['val_NLL']) for r in test_results]

        print('AVERAGE RESULTS OVER ALL SEEDS:')
        print(f'test_NLL = {np.round(np.mean(test_NLL),6)} +/- {np.round(np.std(test_NLL),6)}, test_accuracy = {np.round(np.mean(test_accuracy),6)} +/- {np.round(np.std(test_accuracy),6)} test_brier = {np.round(np.mean(test_brier),6)} +/- {np.round(np.std(test_brier),6)}, test_entropy = {np.round(np.mean(test_entropy),6)} +/- {np.round(np.std(test_entropy),6)}, test_auroc = {np.round(np.mean(test_auroc),6)} +/- {np.round(np.std(test_auroc),6)}')
        
    else:
        test_NLL, test_MSE, test_RMSE = [],[],[]
        for r in test_results:
            print(r)
            test_NLL.append(r['NLL'])
            test_MSE.append(r['MSE'])
            test_RMSE.append(r['RMSE'])
        print('VALIDATION NLL...\n')
        [print(r['val_NLL']) for r in test_results]
        print('AVERAGE RESULTS OVER ALL SEEDS:')
        print(f'test_NLL = {np.round(np.mean(test_NLL),6)} +/- {np.round(np.std(test_NLL),6)}, test_MSE = {np.round(np.mean(test_MSE),6)} +/- {np.round(np.std(test_MSE),6)}, test_RMSE = {np.round(np.mean(test_RMSE),6)} +/- {np.round(np.std(test_RMSE),6)}')




    


    

if __name__ == "__main__":
    main()
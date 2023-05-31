import torch
import argparse

from utils.save_results import save_results
from utils.generate_data import generate_data
from utils.baselines.detectGPT import detectGPT
from utils.baselines.run_baselines import run_baselines
from utils.setting import set_experiment_config, initial_setup
from utils.load_models_tokenizers import load_base_model_and_tokenizer, load_base_model, load_mask_filling_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum", help = "The dataset you want to run your experiments on. Natively supported: XSum, PubMedQA, WritingPrompts, SQuAD, English and German splits of WMT16.")
    parser.add_argument('--dataset_key', type=str, default="document", help = "The column of the dataset you want to use. For instance, the original experiments were ran on the 'document' column of the XSum dataset.")
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2, help = "Span lengths to mask for the mask filling model. A value of 2 performs the best.")
    parser.add_argument('--n_samples', type=int, default=200, help = "Number of samples to run the experiment on. For eg, if set to 200, will only use 200 and run all the experiments on them.")
    parser.add_argument('--n_perturbation_list', type=str, default="1,10", help = "Number of perturbed texts to generate to approximate the expectation term in the eq. 1 of the paper.")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1, help = "Rounds of perturbations to apply.")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium", help = "Base model to use to generate the machine-generated text.")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large", help = "Model to use for filling the masks.")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40, help = "Used for decoding strategy while generating the machine-generated text.")
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96, help = "Used for decoding strategy while generating the machine-generated text.")
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--DEVICE', type=str, default ='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    config = {}

    # Parsing the command line arguments and setting up the experiment
    initial_setup(args, config)
    set_experiment_config(args, config)

    # Loading the relevant models (base model, mask filling model) and the corresponding tokenizers.
    load_base_model_and_tokenizer(args, config, None)
    load_mask_filling_model(args, config)

    # Moving the model to DEVICE.
    load_base_model(args, config)

    # Reading original data and generating machine-generated data.
    data = generate_data(args, config)

    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del config["base_model"]
        del config["base_tokenizer"]
        torch.cuda.empty_cache()
        load_base_model_and_tokenizer(args, config, args.scoring_model_name)
        load_base_model(args, config)  # Load again because we've deleted/replaced the old model

    # Running the baselines - rank, logrank, likelihood, entropy.
    if not args.skip_baselines:
        baseline_outputs = run_baselines(args, config, data)
    
    # Running the proposed algorithm.
    outputs = []
    if not args.baselines_only:
        outputs = detectGPT(args, config, data)

    # Saving the results.
    save_results(args, config, baseline_outputs, outputs)
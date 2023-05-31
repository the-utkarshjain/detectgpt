
# DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

## Refactored implementation of the experiments in the [DetectGPT paper](https://arxiv.org/abs/2301.11305v1).

An interactive demo of DetectGPT can be found [here](https://detectgpt.ericmitchell.ai).

## Instructions
1. First, install the Python dependencies: <br/>
    `$ python3 -m venv env` <br/>
    `$ source env/bin/activate` <br/>
    `$ pip install -r requirements.txt` <br/>
  
2. Second, run main.py file (or any of the scripts in `paper_scripts/`): <br/>
`python run.py --base_model_name gpt2 --mask_filling_model_name t5-small --DEVICE cuda`

3. If you'd like to run the WritingPrompts experiments, you'll need to download the WritingPrompts data from [here](https://www.kaggle.com/datasets/ratthachat/writing-prompts). Save the data into a directory `data/writingPrompts`.
 

**Note: Intermediate results are saved in `tmp_results/`. If your experiment completes successfully, the results will be moved into the `results/` directory.**

## Interpreting the results
Once you successfully run the script, the results will be saved in the `results/` directory. The program generates various files and here is their description:

1. **args.json** <br/>
This contains the command line arguments that were passed when the `main.py` file was run.
2. **entropy_threshold_results.json** <br/>
This file has several fields in it. The *prediction* field holds the entropy values of the real text and the sampled text. If you set *n_samples* to 200, each of the subfields (i.e ["predictions"]["real"] and  ["predictions"]["samples"]) will contain 200 values. The *raw_results* field is a list of *n_samples* dictionary where each dictionary contains the original text and its corresponding entropy value (denoted by *original_crit*), the sampled text and its corresponding entropy value (denoted by *sampled_crit*). There are some more fields towards the end of the file (for eg.  *metrics*, *pr_metrics*, etc) but I don't know what they mean.
3. **likelihood_threshold_results.json** <br/>
This file has exactly the same structure but now the values are mean log likelihoods.
4. **rank_threshold_results.json** <br/>
Same as above, but now the values are the negative rank values.
5. **logrank_threshold_results.json** <br/>
Same as above, but now the values are negative logrank values.

Finally, based on the values you select for *n_perturbation_list*, you will have more files. For eg. if you set *n_perturbation_list* to 1,10, you will have four more files:

 1. **perturbation_1_d_results.json** <br/>
	* The ["prediction"]["real"] field stores the unnormalized perturbation discrepancy for the real text (written by human) with just one (i.e *k=1*) perturbed original text to approximate the expectation term in eq 1 of the paper). The ["prediction"]["samples"] field stores the same thing but for the machine generated text. Each of the ["prediction"]["real"] and ["prediction"]["samples"] should contain *n_samples* number of entries.
	* The *raw_results* field contain *n_samples* dictionaries. Each dictionary holds the original text (*original*) and its original loglikelihood (*original_ll*), perturbed original text (*perturbed_original*; should be a list with just one string because we are only using one string to approximate the expectation term) and its loglikelihood (*all_perturbed_original_ll*; should be a list with just one entry), machine generated sample (*sampled*) and its loglikelihood (*sampled_ll*), perturbed sample (*perturbed_sampled*; again, should be a list on just one string) and its loglikelihood (*all_perturbed_sampled_ll*; again, should be a list with just one entry). 
	* The *perturbed_original_ll* holds the mean of the *all_perturbed_original_ll* list and the *perturbed_original_ll_std* holds the standard deviation. Same goes for *perturbed_sampled_ll* and *perturbed_sampled_ll_std*.

2. **perturbation_1_z_results.json** <br/>
Exactly same as above but the ["prediction"]["real"] and ["prediction"]["samples"] now store the **normalized perturbation discrepancy** values.

3. **perturbation_10_d_results.json** <br/>
Contains unnormalized perturbation discrepancy values. The only difference is that since we are using 10 perturbed samples to approximate the expectation term in eq. 1 of the paper, *perturbed_original*, *all_perturbed_original_ll*, *perturbed_sampled*, and *all_perturbed_sampled_ll* should now contain 10 values each.

4. **perturbation_10_z_results.json** <br/>
Same as above but contains **normalized perturbation discrepancy** values.


## Citing the paper

If our work is useful for your own, you can cite us with the following BibTex entry:

@misc{mitchell2023detectgpt,

url = {https://arxiv.org/abs/2301.11305},

author = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and Manning, Christopher D. and Finn, Chelsea},

title = {DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},

publisher = {arXiv},

year = {2023},

}

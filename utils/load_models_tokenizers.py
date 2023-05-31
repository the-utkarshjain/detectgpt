import time
import torch
import transformers

def load_mask_model(args, config):
    mask_model = config["mask_model"]
    base_model = config["base_model"]
    DEVICE = args.DEVICE

    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

    config["mask_model"] = mask_model
    config["base_model"] = base_model
    

def load_base_model(args, config):
    mask_model = config["mask_model"]
    base_model = config["base_model"]
    DEVICE = args.DEVICE

    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

    config["mask_model"] = mask_model
    config["base_model"] = base_model


def load_base_model_and_tokenizer(args, config, model_name = None):
    """
    Loads the base model and corresponding tokenizer based on the input given during runtime.
    """
    if model_name is not None:
        name = model_name
    else:
        name = args.base_model_name
        
    cache_dir = args.cache_dir

    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir = cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    config["base_model"] = base_model
    config["base_tokenizer"] = base_tokenizer

    # this is used if args.openai_model is not none.
    config["GPT2_TOKENIZER"] = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir = cache_dir)


def load_mask_filling_model(args, config):
    """
    Loads the mask filling t5 model for running the perturbation experiments
    """
    mask_filling_model_name = args.mask_filling_model_name
    cache_dir = args.cache_dir

    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir = cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512

    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir = cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir = cache_dir)

    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer
    
    config["mask_model"] = mask_model
    config["preproc_tokenizer"] = preproc_tokenizer
    config["mask_tokenizer"] = mask_tokenizer
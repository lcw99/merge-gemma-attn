import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
import copy

def merge_weights(pretrained_model1, pretrained_model2, larger_model):
    for (pretrained_name1, pretrained_param1), (pretrained_name2, pretrained_param2), (larger_name, larger_param) in zip(
        pretrained_model1.named_parameters(), pretrained_model2.named_parameters(), larger_model.model.named_parameters()
    ):
        # print(f"{pretrained_name}, {larger_param.data.shape=}, {pretrained_param.data.shape=}")
        expand_int = larger_model.config.num_attention_heads // pretrained_model1.config.num_attention_heads
        if "q_proj" in pretrained_name1 or "k_proj" in pretrained_name1 or "v_proj" in pretrained_name1:
            larger_param.data.copy_(torch.cat([pretrained_param1.data, pretrained_param2.data], 0))
        elif "o_proj" in pretrained_name1:
            larger_param.data.copy_(torch.cat([pretrained_param1.data, pretrained_param2.data], -1))
        else: 
            larger_param.data.copy_(pretrained_param1.data)


def build_larger_self_attn_model_with_merge(model1, model2, multiple):
    print(f"loading {model1=}")
    pretrained_model1 = AutoModelForCausalLM.from_pretrained(model1)
    print(f"loading {model2=}")
    pretrained_model2 = AutoModelForCausalLM.from_pretrained(model2)

    larger_config = copy.deepcopy(pretrained_model1.config)
    larger_config.num_attention_heads = pretrained_model1.config.num_attention_heads * multiple
    larger_config.num_key_value_heads = pretrained_model1.config.num_key_value_heads * multiple

    larger_model = AutoModelForCausalLM.from_config(larger_config)

    merge_weights(pretrained_model1, pretrained_model2, larger_model)
    
    return pretrained_model1, larger_model

if __name__ == "__main__":
    DESCRIPTION = """
    enlarge self attention
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--model1",
        type=str,
        default="google/gemma-1.1-7b-it",
        help="path to the model",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="beomi/gemma-ko-7b",
        help="path to the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./merged-output",
        help="output dir",
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=2,
        help="multipe of self attention",
    )
    args = parser.parse_args()
    model1 = args.model1
    model2 = args.model2
    output_model_path = args.output
    multiple = args.multiple

    tokenizer = AutoTokenizer.from_pretrained(model1)

    source_model, larger_model = build_larger_self_attn_model_with_merge(model1, model2, multiple)

    # save enlarged model with original torch_dtype
    tokenizer.save_pretrained(output_model_path)
    larger_model.model.to(source_model.config.torch_dtype)
    larger_model.model.save_pretrained(output_model_path, safe_serialization=True)
    


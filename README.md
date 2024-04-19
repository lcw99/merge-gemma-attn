# Gemma Self-Attention Merger

This script merges the self-attention layers of two pre-trained Gemma 7B models, one English-based and one Korean-based, to create a larger model with increased self-attention capacity.

## Overview

The goal of this project is to leverage the capabilities of both the English and Korean Gemma 7B models by combining their self-attention layers. The resulting merged model can potentially perform better on tasks involving both English and Korean text, as it inherits the strengths of both source models.

## Features

- Merges the self-attention layers of two pre-trained Gemma 7B models
- Doubles the number of attention heads in the merged model
- Allows for the creation of a more versatile and capable language model

## Usage

    python merge_self_attn_head.py --model1 [path_to_english_model] --model2 [path_to_korean_model] --output [output_directory] --multiple 2

Replace the following:
- `[path_to_english_model]`: The path to the English-based Gemma 7B model
- `[path_to_korean_model]`: The path to the Korean-based Gemma 7B model
- `[output_directory]`: The directory where the merged model will be saved
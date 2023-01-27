#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Set and load pretrained language models configurations for prompt engineering.
"""


from statistics import mode
from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from openprompt.plms.lm import LMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, \
                         OPTConfig, OPTForCausalLM, \
                         ElectraConfig, ElectraForMaskedLM, ElectraTokenizer, \
                         DistilBertConfig, DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM, \
                         DebertaV2Config, DebertaV2Tokenizer, DebertaV2Model, DebertaV2ForMaskedLM
from collections import namedtuple
from yacs.config import CfgNode
from openprompt.utils.logging import logger


ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'distilbert': ModelClass(**{
        'config': DistilBertConfig,
        'tokenizer': DistilBertTokenizer,
        'model': DistilBertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'deberta': ModelClass(**{
        'config': DebertaV2Config,
        'tokenizer': DebertaV2Tokenizer,
        'model': DebertaV2ForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5TokenizerWrapper
    }),
    't5-lm':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5LMTokenizerWrapper,
    }),
    'opt': ModelClass(**{
        'config': OPTConfig,
        'tokenizer': GPT2Tokenizer,
        'model': OPTForCausalLM,
        'wrapper': LMTokenizerWrapper,
    }),
    'electra': ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model': ElectraForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
}

def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def load_plm(model_name, model_path, specials_to_add = None):
    """A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ["<pad>"]
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config, wrapper

def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    """add the special_tokens to tokenizer if the special token
    is not in the tokenizer.
    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.
    Returns:
        The resized model, The tokenizer with the added special tokens.
    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer

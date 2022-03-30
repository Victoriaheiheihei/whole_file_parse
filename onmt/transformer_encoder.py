"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward
from onmt.bert import BertEmbedding
from onmt.dropout import IndependentDropout
from torch.nn.init import xavier_uniform_
import torch


class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, mask):
    input_norm = self.att_layer_norm(inputs)
    outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                mask=mask)
    inputs = self.dropout(outputs) + inputs
    
    input_norm = self.ffn_layer_norm(inputs)
    outputs = self.feed_forward(input_norm)
    inputs = outputs + inputs
    return inputs


class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings):
    super(TransformerEncoder, self).__init__()

    self.num_layers = num_layers
    self.embeddings = embeddings
    self.bert_embeddings = BertEmbedding("bert-base-chinese", n_layers=12, n_out=d_model) # 这个参数n_layers你看着�?
    #self.embed_dropout = IndependentDropout(p=0.1) # 这个参数p你看着�?
    #self.linear = nn.Linear(in_features=2 * d_model, out_features=d_model, bias=False)

    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def reset_parameters(self, model_opt):
    init_modules = [self.embeddings,
                    #self.linear,
                    self.transformer,
                    self.layer_norm]
    for m in init_modules:
      for p in m.parameters():
        if model_opt.param_init != 0.0:
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot and p.dim() > 1:
              xavier_uniform_(p)

  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src, bert_src, lengths=None):
    """ See :obj:`EncoderBase.forward()`"""
    self._check_args(src, lengths)

    out = self.bert_embeddings(bert_src)  # [batch_size, seq_len, n_out]
    padding_idx = 0
    mask = bert_src.squeeze(-1).eq(padding_idx).unsqueeze(1)  # [B, 1, T]

    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, mask)
    out = self.layer_norm(out)

    return out, out.transpose(0, 1).contiguous(), lengths


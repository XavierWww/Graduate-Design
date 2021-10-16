"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss


def build_loss_compute(model, fields, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.lambda_coverage != 0: 
        assert opt.coverage_attn, "--coverage_attn needs to be set in order to use --lambda_coverage != 0" 

    if opt.coref_vocab or opt.coref_attn:
        #print("Use CorefGeneratorLossCompute")
        compute = onmt.modules.CorefGeneratorLossCompute(
            model.generator, fields['tgt'].vocab, fields['coref_tgt'].vocab,
            opt.copy_attn_force,
            opt.copy_loss_by_seqlength,
            opt.coref_vocab, opt.lambda_coref_vocab,
            opt.coref_attn, opt.lambda_coref_attn,
            opt.flow, opt.lambda_flow,
            opt.flow_history, opt.lambda_flow_history,
            opt.coref_confscore, opt.lambda_coverage, opt.lambda_coverage2
        )
    elif opt.copy_attn:
        print("Use CopyGeneratorLossCompute")
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, fields['tgt'].vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength, lambda_coverage=opt.lambda_coverage)
    else:
        # compute = NMTLossCompute(
        #     model.generator, tgt_vocab,
        #     label_smoothing=opt.label_smoothing if train else 0.0)
        compute = S2SLossCompute(model.generator, fields['tgt'].vocab)
    compute.to(device)

    return compute


class S2SLossCompute(nn.Module):
    """
    Simple loss compute for seq2seq, do not use shards
    Helps to understand the original code:
    https://github.com/OpenNMT/OpenNMT-py/issues/387
    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(S2SLossCompute, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        self.criterion = nn.NLLLoss(
            ignore_index=self.padding_idx, reduction='sum')

    def compute_loss(self, batch, output, attns, normalization):
        target = batch.tgt[1:]
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        loss.div(float(normalization)).backward()

        stats = self._stats(loss.clone(), scores, gtruth)
        return stats

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        target = batch.tgt[1:]
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)
        return stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))
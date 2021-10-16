# class for memory based decoder

from __future__ import division
import torch
import torch.nn as nn

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory
from onmt.decoders.decoder_utils import RNNDecoderState


class MemoryDecoder(nn.Module):
    """Memory decoder to aggregate multiple memories"""
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="mlp", attn_func="softmax",
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, coverage_attn=True):
        super(MemoryDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)
        # attentions
        # self.passage_attn = onmt.modules.MemRationaleAttention(
        #     hidden_size, hidden_size,
        #     attn_type=attn_type, attn_func=attn_func
        # )
        
        #TODO
        self._coverage = coverage_attn
        self.attn = onmt.modules.HierarchicalAttention(
            hidden_size, hidden_size,
            attn_type=attn_type, attn_func=attn_func, coverage=coverage_attn #TODO
        )
        
        '''
            When to use?
        '''
        self.linear_out = nn.Linear(hidden_size * 2,
                                    hidden_size, bias=False)

        #self.linear_out2 = nn.Linear(hidden_size * 1,
        #                            hidden_size, bias=True)

        self._copy = False
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt,
                src_bank, src_lengths,
                qa_sent_bank, qa_sent_lengths,
                qa_word_bank, qa_word_lengths,
                state, step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
                                                    tgt,
                                                    src_bank, src_lengths,
                                                    qa_sent_bank, qa_sent_lengths,
                                                    qa_word_bank, qa_word_lengths,
                                                    state)
        
        #print(attns["copy"])


        # Update the state with the result. 
        final_output = decoder_outputs[-1]
        coverage = None
        coverage2 = None

        state.update_state(decoder_final, final_output.unsqueeze(0), coverage, coverage2)
        
        #TODO
        if "coverage" in attns:
            state.coverage = attns["coverage"][-1].unsqueeze(0)
        if "coverage2" in attns:
            state.coverage2 = attns["coverage2"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs)

            for k in attns:
                if type(attns[k]) == list:
                   attns[k] = torch.stack(attns[k])
                   #print(len(attns[k]))

        return decoder_outputs, state, attns
      
      
    #TODO
    #def update_dropout(self, dropout):
    #    self.dropout.p = dropout
    #    self.embeddings.update_dropout(dropout)


    def _run_forward_pass(self, tgt,
                          src_bank, src_lengths,
                          qa_sent_bank, qa_sent_lengths,
                          qa_word_bank, qa_word_lengths,
                          state):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values:
        
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                ``(len, batch, nfeats)``.
            memory_bank (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(src_len, batch, hidden_size)``.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):
            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """
        
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_length, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": [], "qa": [], "passage": [], "p_cov": [], "qa_cov": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
            attns["coverage2"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        ##TODO
        hidden = state.hidden
#       coverage = None
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None
        coverage2 = state.coverage2.squeeze(0) \
            if state.coverage2 is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, hidden = self.rnn(decoder_input, hidden)  # [batch, hid]

            # # attention over passage
            #passage_context, passage_attn = self.passage_attn(
            #    rnn_output, src_bank.transpose(0, 1).contiguous(), src_lengths
            #)  # [seqlen, batch, hid] -> [batch, seqlen, hid]
            
            ## hierarchical attention over qa
            src_context, src_attn, passage_attn, qa_attn, p_cov_attn, qa_cov_attn = self.attn(
                rnn_output,
                src_bank, src_lengths,
                qa_sent_bank, qa_sent_lengths,
                qa_word_bank, qa_word_lengths,
                coverage, coverage2
            )
            #attns["std"].append(passage_attn)

            # concat and get final represetation
            h_ = torch.cat([src_context, rnn_output], 1)
            decoder_output = torch.tanh(self.linear_out(h_))
            #decoder_output = self.linear_out2(decoder_output)
            decoder_output = self.dropout(decoder_output)
            
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            
            # Update the coverage attention.
            if self._coverage:
                
                '''
                if coverage is None:
                    coverage = 0.85*coverage + 0.15*torch.exp(-p_cov_attn)
                    attns["coverage"] += [coverage]
                if coverage2 is None:
                    coverage2 = 0.85*coverage2 + 0.15*torch.exp(-qa_cov_attn)
                    attns["coverage2"] += [coverage2]
                '''

                #coverage = p_cov_attn if coverage is None else p_cov_attn + coverage
                coverage = p_cov_attn if coverage is None else 0.85*coverage + 0.15*torch.exp(-p_cov_attn)
                attns["coverage"] += [coverage]

                #coverage2 = qa_word_attn if coverage2 is None else qa_word_attn + coverage2
                #attns["coverage2"] += [coverage2]

                #coverage2 = qa_cov_attn if coverage2 is None else qa_cov_attn + coverage2
                coverage2 = qa_cov_attn if coverage2 is None else 0.85*coverage2 + 0.15*torch.exp(-qa_cov_attn)
                attns["coverage2"] += [coverage2]

                
     
            attns["std"] += [src_attn]
            attns["passage"] += [passage_attn]
            attns["qa"] += [qa_attn]
            attns["p_cov"] += [p_cov_attn]
            attns["qa_cov"] += [qa_cov_attn]


            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                # train a separate copy attention layer
                raise NotImplementedError
                '''
                ##TODO
                _, copy_attn = self.copy_attn(
                    decoder_output, src_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
                '''
            elif self._copy:
                attns["copy"] = attns["std"]
                
        # Return result.
        return hidden, decoder_outputs, attns

    ##Don't change
    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

    #TODO
    #def update_dropout(self, dropout):
    #    self.dropout.p = dropout
    #    self.rnn.dropout.p = dropout
    #    self.embeddings.update_dropout(dropout)


    def init_decoder_state(self, encoder_final):
        """ Init decoder state with last state of the rationale encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

        # Init the input feed.
        #batch_size = state.hidden[0].size(1)
        #h_size = (batch_size, self.hidden_size)
        #state.input_feed = \
        #    state.hidden[0].data.new(*h_size).zero_().unsqueeze(0)
        #state.coverage = True


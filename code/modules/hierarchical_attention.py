""" Hierarchical attention modules """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd

class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention takes two matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, indim, outdim, coverage=True, attn_type="mlp",
                 attn_func="softmax"):
        super(HierarchicalAttention, self).__init__()

        self.indim = indim
        self.outdim = outdim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        #print(self.attn_type)

        # Hierarchical attention
        '''
        if self.attn_type == "general":
            self.pass_linear_in = nn.Linear(indim, outdim, bias=False)
            self.qa_word_linear_in = nn.Linear(indim, outdim, bias=False)
            self.qa_sent_linear_in = nn.Linear(indim, outdim, bias=False)
            # self.sent_linear_in = nn.Linear(indim, outdim, bias=False)
        '''

        #TODO
        if self.attn_type == "mlp":
            self.qa_word_linear_context = nn.Linear(indim, outdim, bias=False)
            self.qa_sent_linear_context = nn.Linear(indim, outdim, bias=False)
            self.pass_linear_context = nn.Linear(indim, outdim, bias=False)
            self.qa_word_linear_query = nn.Linear(indim, outdim, bias=True)
            self.qa_sent_linear_query = nn.Linear(indim, outdim, bias=True)
            self.pass_linear_query = nn.Linear(indim, outdim, bias=True)
            self.qa_word_v = nn.Linear(indim, 1, bias=False)
            self.qa_sent_v = nn.Linear(indim, 1, bias=False)
            self.pass_v = nn.Linear(indim, 1, bias=False)
        # mlp wants it with bias
        #out_bias = self.attn_type == "mlp"
        #self.linear_out = nn.Linear(indim * 2, outdim, bias=False)
        
        #TODO
        if coverage:
            self.linear_cover = nn.Linear(1, outdim, bias=False)
            self.linear_cover2 = nn.Linear(1, outdim, bias=False)
            #self.linear_cover3 = nn.Linear(1, outdim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def score(self, h_t, h_s, type):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          type: use word or sent matrix
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        
        #TODO
        aeq(self.indim, src_dim)
        aeq(self.outdim, src_dim)
        
        '''
        if self.attn_type == "general":
            h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
            if type == 'qa_word':
                h_t_ = self.qa_word_linear_in(h_t_)
            elif type == 'qa_sent':
                h_t_ = self.qa_sent_linear_in(h_t_)
            elif type == 'pass':
                h_t_ = self.pass_linear_in(h_t_)
                
            h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        '''

        if self.attn_type == "mlp":
            #TODO: attn_type='mlp'
            #dim = self.indim #outdim
            if type == 'qa_word':
                wq = self.qa_word_linear_query(h_t.view(-1, tgt_dim))
                uh = self.qa_word_linear_context(h_s.contiguous().view(-1, tgt_dim))
            elif type == 'qa_sent':
                wq = self.qa_sent_linear_query(h_t.view(-1, tgt_dim))
                uh = self.qa_sent_linear_context(h_s.contiguous().view(-1, tgt_dim))
            elif type == 'pass':
                wq = self.pass_linear_query(h_t.view(-1, tgt_dim))
                uh = self.pass_linear_context(h_s.contiguous().view(-1, tgt_dim))
            
            #wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, tgt_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, tgt_dim)

            #uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, tgt_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, tgt_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)
            
            if type == 'qa_word':
                return self.qa_word_v(wquh.view(-1, tgt_dim)).view(tgt_batch, tgt_len, src_len)
            elif type == 'qa_sent':
                return self.qa_sent_v(wquh.view(-1, tgt_dim)).view(tgt_batch, tgt_len, src_len)
            elif type == 'pass':
                return self.pass_v(wquh.view(-1, tgt_dim)).view(tgt_batch, tgt_len, src_len)
                
            #return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)  # v need to be different?
        
    def forward(self, source,
                src_bank, src_lengths, # memory_bank, memory_lengths
                qa_sent_bank, qa_sent_lengths,
                qa_word_bank, qa_word_lengths,
                coverage=None, coverage2=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False
            
        # sent_batch == target_batch
        src_max_len, src_batch, src_dim = src_bank.size() #src_bank = memory_bank
        qa_word_max_len, qa_word_batch, qa_words_max_len, qa_word_dim = qa_word_bank.size()
        assert src_batch == qa_word_batch
        assert src_dim == qa_word_dim
        
        qa_sent_max_len, qa_sent_batch, qa_sent_dim = qa_sent_bank.size()
        assert qa_word_batch == qa_sent_batch
        assert qa_words_max_len == qa_sent_max_len
        
        target_batch, target_l, target_dim = source.size()
        assert src_batch == target_batch # src_batch = batch; target_batch = batch_
        
        #TODO
        aeq(src_dim, target_dim) # src_dim = dim; target_dim = dim_
        aeq(self.indim, src_dim) 
        aeq(self.outdim, src_dim)
        
        #print(coverage is None)

        #TODO
        if coverage is not None:
            target_batch, source_l_ = coverage.size()
            aeq(src_batch, target_batch)
            aeq(src_max_len, source_l_)
        
        #TODO
        if coverage is not None:
            cover = coverage.contiguous().view(-1).unsqueeze(1)
            src_bank += self.linear_cover(cover).view_as(src_bank)
            src_bank = torch.tanh(src_bank)
        
        if coverage2 is not None:
            cover2 = coverage2.contiguous().view(-1).unsqueeze(1)
            qa_word_bank += self.linear_cover2(cover2).view_as(qa_word_bank)
            qa_word_bank = torch.tanh(qa_word_bank)

        #if coverage3 is not None:
        #    cover3 = coverage3.view(-1).unsqueeze(1)
        #    qa_sent_bank += self.linear_cover3(cover3).view_as(qa_sent_bank)
        #    qa_sent_bank = torch.tanh(qa_sent_bank)

        ## reshape for compute word score: qa_word_align, src_align, qa_sent_align
        #                                  qa_hier_align =  qa sent * qa word
        # (qa_word_max_len, qa_word_batch, qa_words_max_len, qa_word_dim) -> transpose
        # (qa_word_batch, qa_word_max_len, qa_words_max_len, qa_word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (qa_word_batch, qa_words_max_len, qa_word_max_len, qa_word_dim)
        qa_word_bank = qa_word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            qa_word_batch, qa_words_max_len*qa_word_max_len, qa_word_dim)
        qa_word_align = self.score(source, qa_word_bank, 'qa_word') # One

        # (src_max_len, src_batch, src_dim) -> (src_batch, src_max_len, src_dim)
        src_bank = src_bank.transpose(0, 1).contiguous()
        src_align = self.score(source, src_bank, 'pass') # Two

        # sentence score
        # (qa_sent_batch, target_l, sent_max_len)
        qa_sent_bank = qa_sent_bank.transpose(0, 1).contiguous()
        qa_sent_align = self.score(source, qa_sent_bank, 'qa_sent') # Three

        # hierarchical qa attention: qa_sent * qa_word (One*Three)
        qa_hier_align = (qa_word_align.view(qa_word_batch, target_l, qa_words_max_len, qa_word_max_len) * \
            qa_sent_align.unsqueeze(-1)).view(qa_word_batch, target_l, qa_words_max_len * qa_word_max_len)

        # how many words from passage; from qa
        p_size = src_align.size()[2]
        #qa_size = qa_hier_align.size()[2]

        # concat src with hier bank : src_align + qa_hier_align
        align = torch.cat([src_align, qa_hier_align], -1) 
        
        # mask : qa_mask, src_mask
        qa_mask = sequence_mask(qa_word_lengths.view(-1), max_len=qa_word_max_len).view(
            qa_word_batch, qa_words_max_len * qa_word_max_len).unsqueeze(1)
        src_mask = sequence_mask(src_lengths, max_len=src_max_len).view(
            src_batch, src_max_len).unsqueeze(1)
        mask = torch.cat([src_mask, qa_mask], -1)
        align.masked_fill_(~(mask.cuda()), -float('inf'))
      
        # qa_hier_align for qa coref loss
        qa_hier_align.masked_fill_(~(qa_mask.cuda()), -float('inf'))
        qa_hier_vectors = self.softmax(qa_hier_align)
        # src_align for qa coref loss
        src_align.masked_fill_(~(src_mask.cuda()), -float('inf'))
        src_vectors = self.softmax(src_align)

        #TODO
        #qa_word_mask = sequence_mask()
        #qa_sent_mask = sequence_mask(qa_sent_lengths, max_len=qa_sent_max_len).view(
         #   qa_sent_batch, qa_sent_max_len).unsqueeze(1)
        
        #qa_sent_align.masked_fill_(~(qa_sent_mask.cuda()), -float('inf'))
        #qa_word_vectors = self.softmax(qa_word_align)
        #qa_sent_vectors = self.softmax(qa_sent_align)

        ## normalize attention weights
        align_vectors = self.softmax(align) # (word_batch, target_l, words_max_len * word_max_len)
        #align_vectors = torch.cat([src_vectors, qa_hier_vectors], -1)

        # passage, qa for coverage
        passage_align_vectors = align_vectors[:,:,:p_size]
        qa_align_vectors = align_vectors[:,:,p_size:]

        ## each context vector c_t is the weighted average over all the source hidden states
        memory_bank = torch.cat([src_bank, qa_word_bank], 1) # (word_batch, target_l, hid)
        c = torch.bmm(align_vectors, memory_bank)
        
        '''
        # TODO
        # concatenate
        concat_c = torch.cat([c, source], 2).view(src_batch*target_l, src_dim*2)
        attn_h = self.linear_out(concat_c).view(src_batch, target_l, src_dim)
        
        #if self.attn_type in ["general", "dot"]:
        attn_h = torch.tanh(attn_h)

        
        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            target_batch, target_dim = attn_h.size()
            aeq(src_batch, target_batch)
            aeq(src_dim, target_dim)
            target_batch, source_l_ = align_vectors.size()
            aeq(src_batch, target_batch)
            #aeq(src_max_len, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, target_batch, target_dim = attn_h.size()
            aeq(target_l, target_l_)
            aeq(src_batch, target_batch)
            aeq(src_dim, target_dim)
            target_l_, target_batch, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(src_batch, target_batch)
            #aeq(src_max_len, source_l_)        
        '''
        return c.squeeze(1), align_vectors.squeeze(1), src_vectors.squeeze(1), qa_hier_vectors.squeeze(1), passage_align_vectors.squeeze(1), qa_align_vectors.squeeze(1)
        #return c.squeeze(1), align_vectors.squeeze(1), src_vectors.squeeze(1), qa_hier_vectors.squeeze(1)

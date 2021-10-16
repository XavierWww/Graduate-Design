"""  Attention and normalization modules  """
from onmt.modules.util_class import LayerNorm, Elementwise
#from onmt.modules.gate import context_gate_factory, ContextGate
#from onmt.modules.global_attention import GlobalAttention
#from onmt.modules.conv_multi_step_attention import ConvMultiStepAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLossCompute
#from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding
from onmt.modules.weight_norm import WeightNormConv2d
#from onmt.modules.average_attn import AverageAttention
#from onmt.modules.memory_attention import MemAttention, MemAnswerAttention, \
#    MemQuestionAttention, MemRationaleAttention
#from onmt.modules.query_attention import QueryAttention
from onmt.modules.hierarchical_attention import HierarchicalAttention
from onmt.modules.coref_generator import CorefGenerator, CorefGeneratorLossCompute

"""
__all__ = ["LayerNorm", "Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLossCompute", "MultiHeadedAttention", "Embeddings",
           "PositionalEncoding", "WeightNormConv2d", "AverageAttention",
           "MemAttention", "MemAnswerAttention", "MemQuestionAttention",
           "MemRationaleAttention", "QueryAttention", "HierarchicalAttention",
           "CorefGenerator", "CorefGeneratorLossCompute"]
"""

__all__ = ["LayerNorm", "Elementwise", "CopyGenerator",
           "CopyGeneratorLossCompute", "Embeddings",
           "PositionalEncoding", "WeightNormConv2d", "HierarchicalAttention",
           "CorefGenerator", "CorefGeneratorLossCompute"]
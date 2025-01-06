import math
from torch import nn
from torch.nn import functional as F

from src.types_ import *
from src.surrogate import SurrogateVAE


class SurrogateScorerNoReg(nn.Module):
    def __init__(self, sol_emb_dim, ins_emb_dim, hidden_dims, generator_mid_dim=64, relu=False, **kwargs):
        super(SurrogateScorerNoReg, self).__init__()
        self.sol_emb_dim = sol_emb_dim
        self.ins_emb_dim = ins_emb_dim
        self.hidden_dims = hidden_dims
        self.generator_mid_dim = generator_mid_dim
        self.relu = relu
        self.weights_num = 0
        current_dim = self.sol_emb_dim
        for hidden_dim in self.hidden_dims:
            self.weights_num += current_dim * hidden_dim + hidden_dim
            current_dim = hidden_dim
        self.weights_num += current_dim * 1 + 1

    def forward(self, sol_emb: Tensor, ins_emb: Tensor):
        current_index = 0
        current_dim = self.sol_emb_dim
        current_value = sol_emb
        for index, hidden_dim in enumerate(self.hidden_dims):
            weight = ins_emb[current_index:current_index + current_dim * hidden_dim].view(hidden_dim, current_dim)
            current_index += current_dim * hidden_dim
            bias = ins_emb[current_index:current_index + hidden_dim]
            current_index += hidden_dim
            current_value = F.linear(current_value, weight, bias)
            current_dim = hidden_dim
        weight = ins_emb[current_index:current_index + current_dim].view(1, current_dim)
        current_index = current_index + current_dim
        bias = ins_emb[current_index:current_index + 1]
        current_index += 1
        final_value = F.linear(current_value, weight, bias).flatten(start_dim=0)
        if self.relu:
            final_value = F.relu(final_value)
        return final_value


class SurrogateVAENoReg(SurrogateVAE):
    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 encoder_dims: List[int] = None,
                 decoder_dims: List[int] = None,
                 hidden_dims: List[int] = None,
                 ins_emb_dim: int = 32,
                 scorer_generator_mid_dim: int = 64,
                 out_dim: int = None,
                 **kwargs) -> None:
        super(SurrogateVAE, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim if out_dim else in_dim
        self.latent_dim = latent_dim
        self.ins_emb_dim = ins_emb_dim
        self.scorer_generator_mid_dim = scorer_generator_mid_dim
        self.encoder_dims = [64, 128, 128] if encoder_dims is None else encoder_dims
        self.decoder_dims = [64, 128, 128] if decoder_dims is None else decoder_dims
        self.hidden_dims = [64, 64] if hidden_dims is None else hidden_dims

        # Build Encoder
        self.encoder, self.fc_mu, self.fc_var = self.generate_encoder()

        # Build Decoder
        self.decoder, self.decoder_input, self.final_layer = self.generate_decoder()

        # Build Scorer
        self.scorer = SurrogateScorerNoReg(sol_emb_dim=self.latent_dim,
                                           ins_emb_dim=self.ins_emb_dim,
                                           hidden_dims=self.hidden_dims,
                                           generator_mid_dim=self.scorer_generator_mid_dim
                                           )
        self.add_module(name="Scorer", module=self.scorer)
        self.encoder_params = []
        for name, param in self.named_parameters():
            if "encoder_" == name[:7] or "fc_"==name[:3]:
                if "batches_tracked" not in name:
                    self.encoder_params.append((name, param.size()))


    def forward(self, X: Tensor, ins_emb: Tensor, train: bool = False, **kwargs) -> List[Tensor]:
        scorer_weight = ins_emb[:self.scorer.weights_num]
        current_index = self.scorer.weights_num
        value_dict = {}
        for name, shape in self.encoder_params:
            size = math.prod(shape)
            value_dict[name] = ins_emb[current_index:current_index + size].view(shape)
            current_index += size
        for name, param in self.named_parameters():
            if name in value_dict.keys():
                param.copy_(value_dict[name])
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        s = self.scorer(z if train else mu, scorer_weight)
        return [X, mu, log_var, s]

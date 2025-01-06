from torch import nn
from torch.nn import functional as F

from types_ import *


class SurrogateScorer(nn.Module):
    def __init__(self, sol_emb_dim, ins_emb_dim, hidden_dims, generator_mid_dim=64, relu=False, **kwargs):
        super(SurrogateScorer, self).__init__()
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
        self.weight_generator = nn.Sequential(
            nn.Linear(self.ins_emb_dim, self.generator_mid_dim),
            # nn.BatchNorm1d(self.generator_mid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.generator_mid_dim, self.weights_num),
            # nn.BatchNorm1d(self.weights_num),
            # nn.Sigmoid()
        )
        # self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for hidden_dim in self.hidden_dims)

    def forward(self, sol_emb: Tensor, ins_emb: Tensor):
        weight_params = self.weight_generator(ins_emb.view(1, -1)).view(-1)
        current_index = 0
        current_dim = self.sol_emb_dim
        current_value = sol_emb
        for index, hidden_dim in enumerate(self.hidden_dims):
            weight = weight_params[current_index:current_index + current_dim * hidden_dim].view(hidden_dim, current_dim)
            current_index += current_dim * hidden_dim
            bias = weight_params[current_index:current_index + hidden_dim]
            current_index += hidden_dim
            current_value = F.linear(current_value, weight, bias)
            current_dim = hidden_dim
        weight = weight_params[current_index:current_index + current_dim].view(1, current_dim)
        current_index = current_index + current_dim
        bias = weight_params[current_index:current_index + 1]
        current_index += 1
        final_value = F.linear(current_value, weight, bias).flatten(start_dim=0)
        if self.relu:
            final_value = F.relu(final_value)
        return final_value


class SurrogateVAE(nn.Module):
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
        self.scorer = SurrogateScorer(sol_emb_dim=self.latent_dim,
                                      ins_emb_dim=self.ins_emb_dim,
                                      hidden_dims=self.hidden_dims,
                                      generator_mid_dim=self.scorer_generator_mid_dim
                                      )
        self.add_module(name="Scorer", module=self.scorer)

    def generate_encoder(self):
        modules = []
        in_dim = self.input_dim
        for h_dim in self.encoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim

        fc_mu = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        fc_var = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        encoder = nn.Sequential(*modules)
        return encoder, fc_mu, fc_var

    def generate_decoder(self):
        modules = []
        decoder_input = nn.Linear(self.latent_dim, self.decoder_dims[0])
        for i in range(len(self.decoder_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]),
                    nn.BatchNorm1d(self.decoder_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        decoder = nn.Sequential(*modules)
        final_layer = nn.Sequential(
            nn.Linear(self.decoder_dims[-1], self.output_dim),
            # nn.BatchNorm1d(self.output_dim),
            nn.Hardtanh(min_val=-1, max_val=1)
        )
        return decoder, decoder_input, final_layer

    # def generate_scorer(self):
    #     in_dim = self.latent_dim + self.ins_emb_dim
    #     modules = []
    #     for mlp_dim in self.hidden_dims:
    #         modules.append(
    #             nn.Sequential(
    #                 nn.Linear(in_dim, mlp_dim),
    #                 nn.BatchNorm1d(mlp_dim),
    #                 nn.LeakyReLU()
    #             )
    #         )
    #         in_dim = mlp_dim
    #     modules.append(
    #         nn.Sequential(
    #             nn.Linear(self.hidden_dims[-1], 1),
    #             nn.ReLU(),
    #             nn.Flatten(start_dim=0)
    #         )
    #     )
    #     scorer = nn.Sequential(*modules)
    #     return scorer

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Log variance of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, X: Tensor, ins_emb: Tensor, train: bool = False, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        s = self.scorer(z if train else mu, ins_emb)
        return [self.decode(z), mu, log_var, s]

    def loss(self, X: Tensor, y: Tensor, forward_result: List[Tensor], **kwargs) -> Dict[str, Tensor]:
        recons, mu, log_var, s = forward_result[0], forward_result[1], forward_result[2], forward_result[3]

        performance_loss = F.mse_loss(s, y)
        recons_loss = F.mse_loss(recons, X)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        return {"Reconstruction_Loss": recons_loss, "KLD_Loss": kld_loss, "Performance_Loss": performance_loss}

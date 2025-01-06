import logging

from pytorch_lightning import seed_everything

from src.problem_domain import BaseProblem
from src.surrogate.surrogate_vae import SurrogateVAE
from types_ import *

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)



class SurrogateProblem(BaseProblem):
    def __init__(self, dimension: int = 80, domain: str = None, vae_model: SurrogateVAE = None,
                 vae_weights: Dict = None, ins_emb: Tensor = None, gpu_index=None,
                 sample_batch_num: int = 500, sample_batch_size: int = 20000,
                 reference_score=None, **kwargs):
        super().__init__(**kwargs)
        torch.set_float32_matmul_precision('high')
        self.gpu_index = gpu_index
        self.device = torch.device("cuda") if self.gpu_index is not None else torch.device("cpu")
        self.domain = domain
        self.dimension = dimension
        self.vae = vae_model
        self.vae.load_state_dict(vae_weights)
        self.vae = self.vae.to(self.device)
        self.vae = self.vae.requires_grad_(requires_grad=False)
        self.ins_emb = torch.clone(ins_emb).to(self.device)
        self.vae.eval()
        # self.vae = torch.compile(self.vae)
        self.sample_batch_num = sample_batch_num
        self.sample_batch_size = sample_batch_size
        self.reference_score: Tuple[float, float] = self.sample_and_eval(
            batch_num=self.sample_batch_num,
            batch_size=self.sample_batch_size
        ) if reference_score is None else reference_score

    def update_ins_emb(self, ins_emb: Tensor, reference_score=None):
        self.ins_emb = torch.clone(ins_emb).to(self.device)
        self.reference_score: Tuple[float, float] = self.sample_and_eval(
            batch_num=self.sample_batch_num,
            batch_size=self.sample_batch_size
        ) if reference_score is None else reference_score

    def forward(self, x: Union[NpArray, List[int], Tensor]):
        x = torch.tensor(np.array([x]), dtype=torch.float32).to(self.device)
        x = x * 2 - 1
        return self.vae.forward(x, self.ins_emb, train=False)[3].cpu().detach().numpy()[0]

    def evaluate(self, solution: Union[NpArray, List[int], Tensor]) -> float:
        result = (self.forward(solution) - self.reference_score[0]) / (
                self.reference_score[1] - self.reference_score[0])
        return result

    def sample_and_eval(self, batch_num=100, batch_size=20000):
        seed_everything(1088, True)
        all_max = []
        all_min = []
        for _ in range(batch_num):
            solution = torch.randint(0, 2, (batch_size, self.dimension), dtype=torch.float32, device=self.device)
            solution = (solution * 2 - 1)
            score = self.vae.forward(solution, self.ins_emb, train=False)[3]
            all_max.append(torch.max(score).cpu().detach().numpy())
            all_min.append(torch.min(score).cpu().detach().numpy())
        return np.min(all_min), np.max(all_max)

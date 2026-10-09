import torch.nn.functional as func
import torch, math
import torch.nn as nn
from utils import all_gather_batch_with_grad

import tensorly as tl
from tensorly.decomposition import tucker
tl.set_backend('pytorch') # 设置 tensorly 后端为 pytorch

class HRILLoss(nn.Module):
    def __init__(self, temperature=0.1, 
                 w_synergy=None, 
                 target=0.5, 
                 r = 24,
                 proj_dim=24,
                 cov_target=0.5,
                 w_align=5,
                 proj_seed=42,
                 hoidp_detach=False,
                 use_infonce=True):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8

        self.w_align = w_align
        self.cov_target = cov_target 

        self.w_synergy = w_synergy 
        self.target = target
        self.r = r 
        self.proj_dim = proj_dim
        self.hoidp_detach = hoidp_detach
        self.use_infonce = use_infonce

        self._proj_cache = {} # 投影缓存
        self.proj_seed = proj_seed
        self.proj_generator = torch.Generator(device='cpu').manual_seed(proj_seed)

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature 
        sim_zjj = (z2 @ z2.T) / self.temperature 
        sim_zij = (z1 @ z2.T) / self.temperature 
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        return loss
    
    @staticmethod
    def _off_diag(c: torch.Tensor) -> torch.Tensor:
        n, m = c.shape
        return c.flatten()[:-1].view(n - 1, m + 1)[:, 1:].reshape(n - 1, m)
    
    def alignment(self, z1, z2, target: float = 0.0, eps: float = 1e-4):
        super_z1 = torch.cat(z1, dim=-1) # dim [B, D_total]
        super_z2 = torch.cat(z2, dim=-1)

        # 归一化到单位球，度量余弦相似度
        z1n = func.normalize(super_z1, dim=-1)
        z2n = func.normalize(super_z2, dim=-1)

        # 样本级对齐：让同一条样本在两视图中的表示相似（cos -> 1）
        cos_sim = (z1n * z2n).sum(dim=-1)  # [B]
        align_loss = (1.0 - cos_sim).pow(2).mean()

        # 按维度检查方差，防止所有样本坍缩到一个方向
        z_mean = (z1n + z2n) / 2.0
        dim_var = torch.sqrt(z_mean.var(dim=0, unbiased=False) + eps)  # [D_total]
        var_loss = func.relu(target - dim_var).mean()

        return align_loss + var_loss

    def _project_modalities(self, Z_list, r: int = 24):
        """
        固定随机正交投影，降低维度，便于构建外积张量。
        随机正交投影近似保留几何结构（Johnson–Lindenstrauss性质），计算外积或协同项时维度更可控且稳定复现，相对普通随机矩阵更稳定，列正交使得不同投影维度间尽量不相关，距离/内积的保留更好。
        """
        Ys = []
        for idx, Zi in enumerate(Z_list):
            _, D = Zi.shape
            r_use = min(r, D)
            k = (idx, D, r_use)
            Q = self._proj_cache.get(k, None)
            if Q is None or Q.shape != (D, r_use):
                # 使用一个固定的、由key决定的种子来生成可复现的随机矩阵
                A = torch.randn(D, r_use, generator=self.proj_generator)
                # QR分解得到正交基
                Q_cpu, _ = torch.linalg.qr(A, mode='reduced')
                Q = Q_cpu.contiguous()
                self._proj_cache[k] = Q
            Ys.append(Zi @ Q.to(Zi.device, Zi.dtype))
        return Ys

    def synergy_hoidp_loss(self, Z_list, r: int = 24, target: float = 0.1,
                           ranks: list = None, eps: float = 1e-6):
        """
        HOIDP 代理：对batch内外积张量做 Tucker 分解，最大化“非 rank-1”能量比例。
        Z_list: list of [B,Dm], Z_fused: [B,Dz]
        """
        # 1) 预处理与随机正交投影降维

        Y_list = self._project_modalities(Z_list, r=self.proj_dim)
        n_modes = len(Y_list)
        B = Y_list[0].size(0)

        # 2) 构建batch内张量 T = E_b ⊗_{m} y_m[b]
        # 为避免大内存，逐样本累积
        with torch.cuda.amp.autocast(enabled=False):
            T = None
            for b in range(B):
                # 从 [r1]×[r2]×…×[r_{n+1}] 外积开始
                outer = Y_list[0][b].float()
                for m in range(1, n_modes):
                    outer = torch.einsum('...i,j->...ij', outer, Y_list[m][b].float())
                T = outer if T is None else T + outer
            T = T / B  # 经验均值
            # 3) Tucker 分解（HOSVD 初始化 + HOOI 迭代由 tensorly 完成）
            if ranks is None:
                ranks = [min(r, T.shape[i]) for i in range(n_modes)]
            core, _ = tucker(T, rank=ranks, init='svd', tol=1e-3, n_iter_max=15)

        # 稳定的 rank-1 近似能量：取核心最大绝对元素
        core = core.float()
        core_energy = torch.clamp((core.pow(2)).sum(), min=eps) # core的总能量

        # Explaination: 如果 T 基本是 rank-1，那么分解后能量会集中在 core 的单个元素上，alpha2 ≈ core_energy。
        alpha2 = core.abs().amax().pow(2) # core的最大单点能量，对应rank-1近似

        # 非rank-1能量比例
        # 越接近 0：能量几乎全在一个成分上，接近 rank-1，协同性低。
        # 越接近 1：能量分散在多个成分上，协同性高。
        synergy_ratio = torch.clamp(1.0 - (alpha2 / core_energy), min=eps, max=1.0 - 1e-6)

        # target表示至少有多少比例的能量不在rank-1成分上
        loss = func.relu(target - synergy_ratio)
        return loss

    def forward(self, outputs):
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]

        assert len(z1) == len(z2)
        n_emb = len(z1)

        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        z1_u, z2_u = z1[:prototype], z2[:prototype]
        z1_f, z2_f = z1[prototype], z2[prototype]

        loss_list = []
        total_loss = {}

        # =========== InfoNCE loss ===========
        if self.use_infonce:
            for i in range(len(z1_u)):
                loss1 = self.infonce(z1_u[i], z2_f)
                loss2 = self.infonce(z2_u[i], z1_f)
                infonce_loss = (loss1 + loss2) / 2.
                loss_list.append(infonce_loss)
                total_loss[f'infonce_loss_modality_{i}'] = infonce_loss
        
        # =========== Synergy HOIDP loss ===========
        # if self.w_synergy is not None and self.w_synergy > 0:
        if self.w_synergy is not None and self.w_synergy > 0 and self.hoidp_detach==False:
            hoidp_loss1 = self.synergy_hoidp_loss(z1_u, r=self.r, target=self.target)
            hoidp_loss2 = self.synergy_hoidp_loss(z2_u, r=self.r, target=self.target)
            hoidp_loss = self.w_synergy * (hoidp_loss1 + hoidp_loss2) / 2.
            loss_list.append(hoidp_loss)
            total_loss['hoidp_loss'] = hoidp_loss
        # 如果 hoidp_detach=True，则不计算 hoidp_loss 的梯度，作为一个稳定的正则项
        elif self.w_synergy is not None and self.w_synergy > 0 and self.hoidp_detach==True:
            with torch.no_grad():
                hoidp_loss1 = self.synergy_hoidp_loss(z1_u, r=self.r, target=self.target)
                hoidp_loss2 = self.synergy_hoidp_loss(z2_u, r=self.r, target=self.target)
                hoidp_loss = self.w_synergy * (hoidp_loss1 + hoidp_loss2) / 2.
                loss_list.append(hoidp_loss)
                total_loss['hoidp_loss_detach'] = hoidp_loss
        
        # =========== Alignment loss ===========
        if self.w_align > 0:
            align_loss = self.alignment(z1_u, z2_u, target=self.cov_target) * self.w_align
            loss_list.append(align_loss)
            total_loss['align_loss'] = align_loss
        
        loss = torch.stack(loss_list).sum(dim=0)
        return {"loss": loss, **total_loss}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat

from config import Config


class Skipper(nn.Module):
    """Skip connection module for CA models.

    This module implements a skip connection that adds the input
    directly to the output, facilitating gradient flow and preserving
    information across layers.
    """

    def __init__(self, body) -> None:
        """Initialize the Skipper module."""

        super().__init__()
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Skipper.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying skip connection.
        """
        return x + self.body(x)


class MergedCAModel(nn.Module):
    """Merged Cellular Automata model for multiple competing NCAs.

    This model processes multiple NCAs simultaneously using grouped convolutions,
    allowing efficient parallel computation of updates for all competing NCAs.
    """

    def __init__(
        self,
        n_ncas: int = 2,
        input_dim: int = 16,
        n_hidden_layers: int = 0,
        hidden_dim: int = 64,
        output_dim: int = 16,
        kernel_size: int = 3,
        dropout_chance: float = 0.0,
    ) -> None:
        """Initialize the merged CA model.

        Args:
            n_ncas: Number of competing NCAs.
            input_dim: Dimension of input cell state.
            hidden_dim: Hidden dimension for internal processing.
        """
        super().__init__()
        self.N = n_ncas
        self.C = input_dim
        self.NH = n_hidden_layers
        self.HD = hidden_dim
        self.OC = output_dim
        self.KS = kernel_size
        self.DC = dropout_chance

        self.encode = nn.Sequential(
            nn.Conv3d(
                self.C,
                self.N * self.HD,
                self.KS,
                padding=(self.KS - 1) // 2,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(p=self.DC),
        )
        self.reasoning = nn.Sequential(*[self.mid_conv_block() for _ in range(self.NH)])
        self.compression = nn.Sequential(
            nn.Conv3d(self.N * self.HD, self.N * self.OC, 1, groups=self.N, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merged CA model.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            Updated tensor of same shape as input.

        Note:
        """
        x = self.encode(x)
        x = self.reasoning(x)
        return self.compression(x)

    def mid_conv_block(self):
        return Skipper(
            nn.Sequential(
                nn.Conv3d(
                    self.N * self.HD,
                    self.N * self.HD,
                    self.KS,
                    padding=(self.KS - 1) // 2,
                    groups=self.N,
                    bias=False,
                ),
                nn.GELU(),
                nn.Dropout(p=self.DC),
            )
        )


class CAEntity:
    """Wrapper around the NCA which includes the optimizer.

    This class encapsulates a neural cellular automata model along with its
    optimizer, providing methods for training updates and gradient normalization.
    """

    def __init__(self, config: Config) -> None:
        """Initialize CA entity with model and optimizer.

        Args:
            config: Configuration object containing model and training parameters.
        """
        model_in_dim = (
            config.cell_dim if config.alive_visible else config.cell_wo_alive_dim
        )
        self.model = MergedCAModel(
            config.n_ncas,
            model_in_dim,
            config.n_hidden_layers,
            config.hidden_dim,
            config.cell_wo_alive_dim,
            config.model_kernel_size,
            config.model_dropout_per,
        ).to(config.device)

        # NOTE: This was something to fix a bug, but I should figure this out if I'm doing continuous training
        # Only compile during training
        optimizer_map = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "RMSProp": torch.optim.RMSprop,
            "SGD": torch.optim.SGD,
        }

        optimizer_class = optimizer_map.get(config.optimizer)
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=config.learning_rate
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CA model.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)

    def normalize_grads(self) -> None:
        """Normalize gradients for stable training.

        This made training more stable for the original NCA by normalizing
        each parameter's gradient by its norm.

        NOTE: Changing this to gradient norm clipping for a little bit
        Actually the old one was a bit weird since now all the parameters are tied together.
        """
        # for p in self.model.parameters():
        #     p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def update(self) -> None:
        """Update model parameters using optimizer."""
        self.normalize_grads()
        self.optimizer.step()
        self.optimizer.zero_grad()


class CASunGroup:
    """Wrapper which includes the entire group of NCAs along with sunshine.

    This class manages multiple competing NCAs and implements the competition
    mechanics including attack/defense interactions and territory control.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the CA group with competition mechanics.

        Args:
            config: Configuration object with all training parameters.
        """
        # Initial variables
        self.n_ncas = config.n_ncas
        self.N = self.n_ncas + 1  # Total number of updates including sunshine
        self.batch_size = config.batch_size
        self.device = config.device
        self.total_grid_size = config.total_grid_size
        self.mode = config.mode
        self.alive_visible = config.alive_visible
        self.cell_state_dim = config.cell_state_dim
        self.cell_hidden_dim = config.cell_hidden_dim
        self.softmax_temp = config.softmax_temp

        # Make list of proposed upates and then iterate through to find total strengths
        # This is marking what you are fighting, sunshine is idx 0
        # interactions is a list of tuples where the fst one is attacking and second is defending
        # prevents any self attacking
        self.interactions = torch.nonzero(
            torch.ones(self.N, self.N) - torch.diag(torch.ones(self.N))
        ).to(device=config.device)
        self.I = self.interactions.shape[0]
        self.str_add_idx = torch.zeros(self.I, self.N).to(config.device)
        self.str_add_idx[torch.arange(self.I), torch.arange(self.I) // (self.N - 1)] = 1

        self.cell_dim = config.cell_dim
        self.out_dim = config.cell_wo_alive_dim
        self.ali_idxs = torch.arange(self.N, device=config.device)

        self.cell_idxs = self.N + torch.arange(
            config.cell_wo_alive_dim, device=config.device
        )
        self.state_idxs = self.N + torch.arange(
            config.cell_state_dim, device=config.device
        )
        self.hidden_idxs = (
            self.N
            + config.cell_state_dim
            + torch.arange(config.cell_hidden_dim, device=config.device)
        )

        self.att_idxs = torch.arange(self.cell_state_dim // 2, device=config.device)
        self.def_idxs = self.att_idxs + self.cell_state_dim // 2

        # Initialize N NCAs
        self.models = CAEntity(config)

        # NOTE: Maybe try something with softmax temp rather than this aliveness threshold
        self.threshold = torch.tensor(0.4, device=config.device)
        self.perspective_mask = torch.eye(self.N, device=self.device)
        self.perspective_mask = rearrange(
            self.perspective_mask, "n1 n2 -> n1 n2 () () () () ()"
        )

        self.per_hid_upd = config.per_hid_upd

        # TRACKING STATS
        self.inter = None
        # Pre-compute sun update
        self._setup_sun_update(config)

    def _setup_sun_update(self, config: Config) -> None:
        """Setup the sun update vector for baseline competition.

        The sun acts as a neutral background force that competes with NCAs
        for territory control.

        Args:
            config: Configuration object.
        """
        sun_vec = torch.randn(self.out_dim, device=config.device)

        sun_vec[self.hidden_idxs - self.N] = 0.0
        sun_vec /= sun_vec.norm()

        self.sun_update = rearrange(sun_vec, "oc -> () oc () () ()")
        self.sun_update.requires_grad = True

        self.sun_optim = torch.optim.AdamW([self.sun_update], lr=config.learning_rate)

    def _update_sun(self, step: bool = True):
        if step:
            self.sun_optim.step()
        self.sun_update.grad = torch.zeros_like(self.sun_update)

    def _parallel_forward_step(self, x_perspectives: torch.Tensor) -> torch.Tensor:
        """Single forward step for all perspectives in parallel.

        Args:
            x_perspectives: Individual perspectives [N, B, C, D, H, W]
                          where each should only have gradients for NCA ni.

        Returns:
            Updated perspectives tensor of same shape.
        """
        N, B, C, D, H, W = x_perspectives.shape

        x_flat = rearrange(x_perspectives, "n b c d h w -> (n b) c d h w")
        if not self.alive_visible:
            x_flat = x_flat[:, self.cell_idxs]

        hids_visible_to_each = int(self.per_hid_upd * self.cell_hidden_dim)
        xs = torch.arange(self.n_ncas, device=self.device)
        offsets = torch.arange(hids_visible_to_each, device=self.device)
        ys = (xs[:, None] + offsets[None, :]) % self.cell_hidden_dim
        ys = ys.flatten()
        xs = xs.repeat_interleave(hids_visible_to_each)
        vis_grid = torch.zeros(
            self.n_ncas,
            self.cell_state_dim + self.cell_hidden_dim,
            dtype=torch.int8,
            device=self.device,
        )
        vis_grid[:, : self.cell_state_dim] = 1
        vis_grid[xs, ys + self.cell_state_dim] = 1
        vis_grid = rearrange(vis_grid, "n c -> () n () c () () ()")

        all_updates = self.models(x_flat)  # [N*B, OC*N, D, H, W]
        all_updates = rearrange(
            all_updates,
            "(n b) (oc m) d h w -> n m b oc d h w",
            n=self.N,
            m=self.n_ncas,
        )
        all_updates = all_updates * vis_grid

        sun_update = self.sun_update.expand(self.N, 1, B, self.out_dim, D, H, W)
        all_updates = torch.cat([sun_update, all_updates], dim=1)  # [N, M, B, OC, H, W]

        all_updates = all_updates * self.perspective_mask + all_updates.detach() * (
            1 - self.perspective_mask
        )

        x_new = self._run_competition_parallel(x_perspectives, all_updates)

        # NOTE: Different ways of keeping the grid bounded
        # NOTE: :100 was for making sure torch.quantile doesn't break
        x_new = x_new.clamp(-1, 1)
        # x_new = torch.nn.functional.tanh(x_new/torch.quantile(x_new[0,0,:,:100,:100].abs(), 0.9).detach())

        return x_new

    def _run_competition_parallel(
        self, x_perspectives: torch.Tensor, all_updates: torch.Tensor
    ) -> torch.Tensor:
        """Fully parallel competition across all perspectives.

        Args:
            x_perspectives: Input perspectives [N, B, C, D, H, W].
            all_updates: All proposed updates [N, M, B, OC, H, W] where M=N(n_ncas)+1.
            NOTE: In this new one where the sun gets updated as well, N and M are both self.N

        Returns:
            Updated perspectives after competition resolution.
        """
        N, B, C, D, H, W = x_perspectives.shape

        # Since all perspectives are the same, can you just get it for one and that's good enough?
        alive_mask_flat = self._get_nca_alive_mask(x_perspectives[0])  # [M, B, D, H, W]
        alive_mask = repeat(alive_mask_flat, "m b d h w -> n m b d h w", n=N)

        all_updates = all_updates * rearrange(alive_mask, "n m b d h w -> n m b 1 d h w")

        all_attacks = all_updates[:, :, :, self.att_idxs]  # [N, M, B, C_att, D, H, W]
        all_defenses = all_updates[:, :, :, self.def_idxs]  # [N, M, B, C_def, D, H, W]

        att_alive = alive_mask_flat[self.interactions[:, 0]]
        def_alive = alive_mask_flat[self.interactions[:, 1]]
        alive_together = att_alive & def_alive
        pair_alive_together = alive_together.view(self.I, -1)
        touching_ncas = torch.any(pair_alive_together, dim=-1)

        att_idx = self.interactions[touching_ncas, 0]
        def_idx = self.interactions[touching_ncas, 1]

        # cos_sim = F.cosine_similarity(attacks, defenses, dim=3)  # [N, I, B, H, W]
        cos_sim = F.cosine_similarity(
            all_attacks[:, att_idx], all_defenses[:, def_idx], dim=3
        )  # [N, I, B, D, H, W]
        defense_cos_sim = F.cosine_similarity(
            all_attacks[:, def_idx], all_defenses[:, att_idx], dim=3
        )  # [N, I, B, D, H, W]

        if self.mode == "eval":
            cos_sim_reduced = reduce(cos_sim.detach(), "n i b d h w -> i", "mean")
            self.inter = torch.zeros(self.N, self.N, device=self.device)
            self.inter[
                self.interactions[touching_ncas, 0], self.interactions[touching_ncas, 1]
            ] = cos_sim_reduced

        strengths = einsum(
            cos_sim, self.str_add_idx[touching_ncas], "n i b d h w, i m -> n m b d h w"
        ) - einsum(
            defense_cos_sim,
            self.str_add_idx[touching_ncas],
            "n i b d h w, i m -> n m b d h w",
        )

        # Apply alive mask to this!
        strengths = strengths.masked_fill(~alive_mask, -torch.inf)
        strengths = strengths.softmax(dim=1)

        x_new = torch.zeros_like(x_perspectives)
        x_new[:, :, self.cell_idxs] = x_perspectives[:, :, self.cell_idxs] + einsum(
            all_updates, strengths, "n m b c d h w, n m b d h w -> n b c d h w"
        )

        x_new[:, :, self.ali_idxs] = rearrange(strengths, "n m b d h w -> n b m d h w").to(
            x_new.dtype
        )

        x_new[:, :, self.ali_idxs] = torch.where(
            rearrange(alive_mask, "n m b d h w -> n b m d h w"),
            x_new[:, :, self.ali_idxs],
            -torch.inf,
        )

        x_new[:, :, self.ali_idxs] = torch.softmax(
            x_new[:, :, self.ali_idxs] / self.softmax_temp, dim=2
        ).to(x_new.dtype)

        alive_mask_flat = self._get_nca_alive_mask(x_new[0])  # [M, B, D, H, W]
        alive_mask = repeat(alive_mask_flat, "m b d h w -> n b m d h w", n=N)

        # Kill off anything not alive enough
        # You need to accomplish some baseline before being able to stay at some cell
        x_new[:, :, self.ali_idxs] = x_new[:, :, self.ali_idxs] * alive_mask

        # Distribute the remaining aliveness so that it sums to 1
        x_new[:, :, self.ali_idxs] = x_new[:, :, self.ali_idxs] / (
            reduce(x_new[:, :, self.ali_idxs], "n b c d h w -> n b 1 d h w", "sum")
        ).to(x_new.dtype)
        return x_new

    def __call__(
        self, x: torch.Tensor, steps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
        """Run multiple forward steps while maintaining gradient isolation.

        Args:
            x: Input tensor [B, C, D, H, W].
            steps: Number of steps to run.

        Returns:
            Tuple containing:
            - x_perspectives: Perspective grids [N, B, C, D, H, W]
            - x_merged: Merged grid [B, C, D, H, W]
            - inter: Interaction statistics (currently None)
        """
        x_perspectives = repeat(x, "b c d h w -> n b c d h w", n=self.N).clone()

        all_xs = torch.zeros((steps, *x.shape), device=x.device, dtype=x.dtype)

        for s in range(steps):
            x_perspectives = self._parallel_forward_step(x_perspectives)
            all_xs[s].copy_(x_perspectives[0].detach())

        return x_perspectives, all_xs, self.inter

    def _get_nca_alive_mask(self, x_perspectives: torch.Tensor) -> torch.Tensor:
        """Calculate alive mask for all perspectives.

        Aliveness is determined by having more than the threshold amount in a
        cell or its 3x3 neighborhood (using max pooling).

        Args:
            x_perspectives: Input tensor [N*B, C, D, H, W].

        Returns:
            Boolean mask [M, N*B, D, H, W] where M=N+1, indicating alive cells.
        """

        NB, C, D, H, W = x_perspectives.shape

        alive_channels = x_perspectives[:, self.ali_idxs]  # [NB, M, D, H, W]
        alive_flat = rearrange(alive_channels, "nb m d h w -> (nb m) 1 d h w")
        alive_pooled = F.max_pool3d(alive_flat, 3, stride=1, padding=1)
        alive_mask = (
            rearrange(alive_pooled, "(nb m) 1 d h w -> m nb d h w", m=self.N)
            > self.threshold
        )

        # Set the sun to alive everywhere
        alive_mask[0] = True

        return alive_mask

    def update_models(
        self, x_perspectives: torch.Tensor, update_sun: bool = False
    ) -> dict[str, list[float] | list[Any]]:
        """Calculate gradients for each NCA and update them.

        This method computes loss based on territory coverage, backpropagates
        gradients, and updates model parameters.

        Args:
            x_perspectives: Perspective grids [N, B, C, D, H, W]. (includes sun perspective)

        Returns:
            Dictionary containing training statistics:
            - growth: List of growth percentages for sun and each NCA
            - grad_norms: List of gradient norms for monitoring
        """

        M, B, C, D, H, W = x_perspectives.shape
        N = M - 1  # Number of NCAs (excluding sun)

        m_idxs = torch.arange(M, device=self.device)
        n_idxs = torch.arange(N, device=self.device)
        # alivenesses = x_perspectives[n_idxs, :, n_idxs + 1]  # [N, B, H, W]

        # NOTE: This is changed now since we have the sun as a learnable param as well
        # alivenesses = x_perspectives[n_idxs+1, :, n_idxs+1]  # [N, B, H, W]
        # alivenesses = x_perspectives[n_idxs, :, n_idxs]  # [N, B, H, W]
        alivenesses = x_perspectives[m_idxs, :, m_idxs]  # [M, B, H, W]

        # ----------- Hypercycle -------------
        # nca_alivenesses = alivenesses[1:] # [N, B, H, W]
        # next_nca_aliveness = torch.roll(nca_alivenesses, shifts=-1, dims=0)  # [N, B, H, W]

        # x_flat = x_perspectives[n_idxs].view(N * B, C, H, W)
        # alive_mask = self._get_nca_alive_mask(x_flat)  # [M, N*B, H, W]
        # alive_mask = alive_mask.view(M, N, B, H, W) # [M, N, B, H, W]
        # nca_alive_masks = alive_mask[n_idxs + 1, n_idxs] # [N, B, H, W]

        # hypercycle_factor = 1.0
        # hypercycle_bonus = torch.where(
        #     nca_alive_masks,
        #     next_nca_aliveness,
        #     torch.zeros_like(next_nca_aliveness)
        # )

        # nca_alivenesses = nca_alivenesses + hypercycle_factor * hypercycle_bonus

        # alivenesses = torch.cat([alivenesses[:1], nca_alivenesses], dim=0)  # [M, B, H, W]
        # alivenesses = alivenesses + torch.where(x_perspectives[n_idxs, :, n_idxs + 1] > 0,
        #     x_perspectives[n_idxs, :, ((n_idxs + 1) % self.n_ncas) + 1],  # [N, B, H, W]
        #     0
        # )
        # ------------------------------------

        # Go down into batch
        batch_alive = alivenesses.view(M, B, -1).sum(-1)  # [M, B]

        log_growth = torch.asinh(batch_alive + 1e-3).mean(1)  # [N]
        ind_losses = -log_growth

        loss = ind_losses.sum()
        loss.backward()

        grad_norm = torch.nn.utils.get_total_norm(self.models.model.parameters())

        self.models.update()
        # Add new param so that you don't always update the sun
        self._update_sun(update_sun)

        # Average it out over the batch
        single_alive = x_perspectives[m_idxs, :, m_idxs].view(M, B, -1).sum(-1)
        percent_covered = single_alive.mean(-1) / self.total_grid_size  # [M]
        percent_covered = percent_covered * 100.0

        # sun_growth = 100.0 - sum(percent_covered)
        return {
            "loss": loss.detach().item(),
            "ind_loss": ind_losses.tolist(),
            # "growth": [sun_growth] + percent_covered.tolist(),
            "growth": percent_covered.tolist(),
            "grad_norm": grad_norm,
        }

    def save(self, config: Config, run_name: str) -> None:
        """Save the models for later inference.

        Args:
            config: Configuration object to save alongside model.
            run_name: Name for the model checkpoint directory.

        Note:
            Creates directory structure: models/{run_name}/
            Saves: config.json, sun.npy, model.pt
        """
        os.mkdir(f"{run_name}")

        config.save(f"{run_name}/config.json")

        np.save(f"{run_name}/sun.npy", self.sun_update.detach().cpu().numpy())

        m = self.models
        torch.save(
            {
                "optim_state_dict": m.optimizer.state_dict(),
                "model_state_dict": m.model.state_dict(),
            },
            f"{run_name}/model.pt",
        )

    def load(self, loc: str) -> bool:
        """Load previous parameters for the CASunGroup.

        Args:
            loc: Directory path containing saved model files.

        Returns:
            True if loading was successful, False otherwise.

        Note:
            Handles torch.compile compatibility by stripping '_orig_mod.' prefixes.
        """
        self.sun_update = torch.from_numpy(np.load(f"{loc}/sun.npy")).to(self.device)

        first_param = next(self.models.model.parameters())
        before_values = first_param.data.clone()[:5]  # First 5 values

        # This is so that when the model is loaded during eval, it isn't compiled (getting weird issues)
        # NOTE: Should look at this more so that you can compile during "eval" too since it is actually just the same
        checkpoint = torch.load(f"{loc}/model.pt")
        if any(
            key.startswith("_orig_mod.")
            for key in checkpoint["model_state_dict"].keys()
        ):
            checkpoint["model_state_dict"] = {
                key.replace("_orig_mod.", ""): value
                for key, value in checkpoint["model_state_dict"].items()
            }

        self.models.model.load_state_dict(checkpoint["model_state_dict"])
        # NOTE: For some reason, this breaks on Metal, but it might work on CUDA?
        # m.model = torch.compile(m.model)

        self.models.optimizer.load_state_dict(checkpoint["optim_state_dict"])

        # Check values after loading
        after_values = first_param.data[:5]
        if torch.allclose(before_values, after_values):
            print("Didn't load weights correctly")
            return False

        return True

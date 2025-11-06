import torch
import torch.nn as nn
from typing import Dict, List, Optional
import ocnn
from ocnn.octree import Octree
# from ...geometry_pr import map_logits_to_angles  # 可选：如果你更想用自己的映射函数，可在 forward 里替换为该函数

class UNet(nn.Module):
    """
    Configurable U-Net for octree features with cutter-aware pose regression.

    Output:
      - (pitch, roll) in degrees
        pitch ∈ [-90, 90], roll ∈ [-180, 180]

    Modes:
      - use_decoder=True: original decoder path (with per-stage cutter fusion) -> Interp -> GAP -> MLP -> 2D
      - use_decoder=False: encoder-only, pyramid pooling on selected depths -> tool fusion -> MLP -> 2D

    Options:
      - tool_fusion: 'concat' (default) or 'film' or 'no_tool'
      - use_attention_pool: False (mean pooling) or True (learnable attention pooling)
      - use_tanh_head: add Tanh on the 2D output (then scaled to angle ranges)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,            # kept for compatibility; not used
        interp: str = 'linear',
        nempty: bool = False,
        *,
        use_decoder: bool = True,
        pyramid_depths: Optional[List[int]] = None,  # only used when use_decoder=False
        tool_fusion: str = 'concat',                 # 'concat' | 'film' | 'no_tool'
        tool_embed_dim: int = 128,
        use_attention_pool: bool = False,
        use_tanh_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nempty = nempty
        self.use_decoder = use_decoder
        self.pyramid_depths = pyramid_depths
        self.tool_fusion = tool_fusion.lower()
        assert self.tool_fusion in ['concat', 'film', 'no_tool']
        self.tool_embed_dim = tool_embed_dim
        self.use_attention_pool = use_attention_pool
        self.use_tanh_head = use_tanh_head

        # ---------------- config ----------------
        self._config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

        # ---------------- encoder ----------------
        self.conv1 = ocnn.modules.OctreeConvBnRelu(
            in_channels, self.encoder_channel[0], nempty=nempty
        )
        self.downsample = nn.ModuleList([
            ocnn.modules.OctreeConvBnRelu(
                self.encoder_channel[i], self.encoder_channel[i + 1],
                kernel_size=[2], stride=2, nempty=nempty
            ) for i in range(self.encoder_stages)
        ])
        self.encoder = nn.ModuleList([
            ocnn.modules.OctreeResBlocks(
                self.encoder_channel[i + 1], self.encoder_channel[i + 1],
                self.encoder_blocks[i], self.bottleneck, nempty, self.resblk
            ) for i in range(self.encoder_stages)
        ])

        # ---------------- decoder (optional) ----------------
        if self.use_decoder:
            # channels after concat: upsampled + skip + tool(256)
            concat_channels = [
                self.decoder_channel[i + 1] + self.encoder_channel[-i - 2]
                for i in range(self.decoder_stages)
            ]
            for k in range(4):
                concat_channels[k] += 256  # add tool features

            self.upsample = nn.ModuleList([
                ocnn.modules.OctreeDeconvBnRelu(
                    self.decoder_channel[i], self.decoder_channel[i + 1],
                    kernel_size=[2], stride=2, nempty=nempty
                ) for i in range(self.decoder_stages)
            ])
            self.decoder = nn.ModuleList([
                ocnn.modules.OctreeResBlocks(
                    concat_channels[i], self.decoder_channel[i + 1],
                    self.decoder_blocks[i], self.bottleneck, nempty, self.resblk
                ) for i in range(self.decoder_stages)
            ])

            # 4× cutter FCs for each decoder stage
            def make_tool_fc():
                return nn.Sequential(
                    nn.Linear(4, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
                    nn.Linear(32, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                )
            self.fc_module_1 = make_tool_fc()
            self.fc_module_2 = make_tool_fc()
            self.fc_module_3 = make_tool_fc()
            self.fc_module_4 = make_tool_fc()

            final_C = self.decoder_channel[-1]  # 96
        else:
            # encoder-only: no decoder; we will pool encoder features
            final_C = self._calc_pyramid_out_channels()

            # tool embedding for late fusion or FiLM
            self.tool_embed = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(inplace=True), nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, self.tool_embed_dim), nn.ReLU(inplace=True),
            )
            
                        # tool embedding for late fusion or FiLM
            # self.tool_embed = nn.Sequential(
            #    nn.Linear(4, 32), nn.ReLU(inplace=True), nn.BatchNorm1d(32), nn.Dropout(0.3),
            #     nn.Linear(32, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.3),
            # )
            

            if self.use_attention_pool:
                self.attn_mlps = nn.ModuleDict()  # lazily created per depth

            if self.tool_fusion == 'film':
                self.film_gamma = None  # lazily created to match final_C
                self.film_beta = None

        # ---------------- common ----------------
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

        # pose head: final_C (+ tool) -> 2 (pitch, roll)
        head_in = final_C
        if not self.use_decoder and self.tool_fusion == 'concat':
            head_in = final_C + self.tool_embed_dim

        head_layers = [
            nn.Linear(head_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # 改为输出 2 维：pitch, roll
        ]
        if self.use_tanh_head:
            head_layers.append(nn.Tanh())  # 输出先约束到 (-1, 1)，再在 forward 里缩放到度
        self.pose_head = nn.Sequential(*head_layers)

        # 小范围初始化最后一层，稳定早期训练
        last_linear = self.pose_head[-2] if self.use_tanh_head else self.pose_head[-1]
        nn.init.uniform_(last_linear.weight, -1e-3, 1e-3)
        nn.init.zeros_(last_linear.bias)

    # ---------------- utilities ----------------
    def _config_network(self):
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.decoder_channel = [256, 256, 128, 96, 96]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2, 2, 2]
        self.head_channel = 64
        self.bottleneck = 1
        self.resblk = ocnn.modules.OctreeResBlock2

    def _calc_pyramid_out_channels(self) -> int:
        if self.pyramid_depths is None:
            return self.encoder_channel[-1]
        return sum(self.encoder_channel)

    def _lazy_build_attn(self, depth: int, in_dim: int, device: torch.device):
        key = str(depth)
        def _make():
            mlp = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            ).to(device)
            return mlp
        if key not in self.attn_mlps:
            self.attn_mlps[key] = _make()
        else:
            cur_in = self.attn_mlps[key][0].in_features
            if cur_in != in_dim:
                self.attn_mlps[key] = _make()

    def _lazy_build_film(self, feat_dim: int, embed_dim: int, device: torch.device):
        if getattr(self, "film_gamma", None) is None or getattr(self, "film_beta", None) is None:
            self.film_gamma = nn.Sequential(
                nn.Linear(embed_dim, feat_dim), nn.Tanh()
            ).to(device)
            self.film_beta = nn.Sequential(
                nn.Linear(embed_dim, feat_dim)
            ).to(device)

    # ---------------- encoder/decoder ----------------
    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):
        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d - 1] = self.encoder[i](conv, octree, d - 1)
        return convd

    def unet_decoder(
        self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int,
        tool_features_1, tool_features_2, tool_features_3, tool_features_4
    ):
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)

            copy_counts = octree.batch_nnum[i + 2]
            expanded_tool_features = []
            if i == 0:
                for j in range(tool_features_1.size(0)):
                    expanded_tool_features.append(tool_features_1[j, :].repeat(copy_counts[j], 1))
            if i == 1:
                for j in range(tool_features_2.size(0)):
                    expanded_tool_features.append(tool_features_2[j, :].repeat(copy_counts[j], 1))
            if i == 2:
                for j in range(tool_features_3.size(0)):
                    expanded_tool_features.append(tool_features_3[j, :].repeat(copy_counts[j], 1))
            if i == 3:
                for j in range(tool_features_4.size(0)):
                    expanded_tool_features.append(tool_features_4[j, :].repeat(copy_counts[j], 1))
            expanded_tool_features = torch.cat(expanded_tool_features, dim=0)

            deconv = torch.cat([expanded_tool_features, deconv], dim=1)
            deconv = torch.cat([convd[d + 1], deconv], dim=1)
            deconv = self.decoder[i](deconv, octree, d + 1)
        return deconv

    # ---------------- pooling helpers ----------------
    @staticmethod
    def _batch_mean_pool(point_feat: torch.Tensor, batch_id: torch.Tensor, B: int):
        C = point_feat.size(1)
        sum_feat = torch.zeros(B, C, device=point_feat.device, dtype=point_feat.dtype)
        sum_feat.index_add_(0, batch_id, point_feat)
        cnt = torch.bincount(batch_id, minlength=B).clamp_min(1).float().to(point_feat.device)
        return sum_feat / cnt.unsqueeze(1)

    def _batch_attn_pool(
        self, point_feat: torch.Tensor, batch_id: torch.Tensor, B: int,
        tool_embed: Optional[torch.Tensor], depth: int
    ):
        in_dim = point_feat.size(1) + (tool_embed.size(1) if tool_embed is not None else 0)
        if not self.use_attention_pool:
            return self._batch_mean_pool(point_feat, batch_id, B)

        self._lazy_build_attn(depth, in_dim, point_feat.device)

        if tool_embed is not None:
            per_point_tool = tool_embed[batch_id]
            attn_in = torch.cat([point_feat, per_point_tool], dim=1)
        else:
            attn_in = point_feat

        scores = self.attn_mlps[str(depth)](attn_in).squeeze(-1)
        scores = scores - scores.detach().max()
        weights = torch.exp(scores)
        sum_w = torch.zeros(B, device=point_feat.device, dtype=weights.dtype)
        sum_w.index_add_(0, batch_id, weights)

        weighted = point_feat * weights.unsqueeze(1)
        pooled = torch.zeros(B, point_feat.size(1), device=point_feat.device, dtype=point_feat.dtype)
        pooled.index_add_(0, batch_id, weighted)

        return pooled / sum_w.clamp_min(1e-6).unsqueeze(1)

    # ---------------- forward ----------------
    def forward(self, data: torch.Tensor, octree: Octree, depth: int,
                query_pts: torch.Tensor, tool_params: torch.Tensor):
        """
        Returns:
            angles: torch.Tensor of shape [B, 2] in degrees, order = [pitch, roll]
            with pitch in [-90, 90], roll in [-180, 180].
        """
        # ----- encoder -----
        convd = self.unet_encoder(data, octree, depth)
        B = tool_params.size(0)

        if self.use_decoder:
            # cutter feature per stage
            tool_features_1 = self.fc_module_1(tool_params)
            tool_features_2 = self.fc_module_2(tool_params)
            tool_features_3 = self.fc_module_3(tool_params)
            tool_features_4 = self.fc_module_4(tool_params)

            # decode
            d_enc = depth - self.encoder_stages
            deconv = self.unet_decoder(
                convd, octree, d_enc,
                tool_features_1, tool_features_2, tool_features_3, tool_features_4
            )

            # interp to points (final decoder feature)
            interp_depth = d_enc + self.decoder_stages
            point_feat = self.octree_interp(deconv, octree, interp_depth, query_pts)  # [N_pts, C=96]

            # global mean by batch
            batch_id = query_pts[:, 3].long()
            global_feat = self._batch_mean_pool(point_feat, batch_id, B)              # [B, 96]

            # head -> 2 logits, then map到角度范围
            raw = self.pose_head(global_feat)  # [B, 2]; 若 use_tanh_head=True 已在 (-1,1)
            if not self.use_tanh_head:
                raw = torch.tanh(raw)
            pitch = 90.0 * raw[:, 0]
            roll  = 180.0 * raw[:, 1]
            angles = torch.stack([pitch, roll], dim=1)
            return angles

        # -------- encoder-only path (no decoder) --------
        tool_params = tool_params.to(next(self.parameters()).device)

        tool_embed = self.tool_embed(tool_params)  # [B, D_tool]

        if self.pyramid_depths is None:
            depths = [depth - self.encoder_stages]  # deepest encoder output
        else:
            depths = self.pyramid_depths

        batch_id = query_pts[:, 3].long()
        pooled_list = []

        for d_i in depths:
            feat_i = convd[d_i]
            pfeat_i = self.octree_interp(feat_i, octree, d_i, query_pts)  # [N_pts, C_i]
            pooled_i = self._batch_attn_pool(
                pfeat_i, batch_id, B, tool_embed if self.use_attention_pool else None, d_i
            )  # [B, C_i]
            pooled_list.append(pooled_i)

        global_feat = torch.cat(pooled_list, dim=1)  # [B, ΣC_i]

        # FiLM or concat fusion
        if self.tool_fusion == 'film':
            self._lazy_build_film(global_feat.size(1), self.tool_embed_dim, global_feat.device)
            gamma = self.film_gamma(tool_embed)     # [B, ΣC_i]
            beta = self.film_beta(tool_embed)       # [B, ΣC_i]
            fused = gamma * global_feat + beta      # [B, ΣC_i]
            raw = self.pose_head(fused)             # [B, 2]
        elif self.tool_fusion == 'no_tool':
            # 不融合刀具参数
            if self.pose_head[0].in_features != global_feat.size(1):
                head_layers = [
                    nn.Linear(global_feat.size(1), 128),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2),
                ]
                if self.use_tanh_head:
                    head_layers.append(nn.Tanh())
                self.pose_head = nn.Sequential(*head_layers)
            raw = self.pose_head(global_feat)       # [B, 2]
        else:  # concat
            fused = torch.cat([global_feat, tool_embed], dim=1)  # [B, ΣC_i + D_tool]
            if self.pose_head[0].in_features != fused.size(1):
                head_layers = [
                    nn.Linear(fused.size(1), 128),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2),
                ]
                if self.use_tanh_head:
                    head_layers.append(nn.Tanh())
                self.pose_head = nn.Sequential(*head_layers)
            raw = self.pose_head(fused)             # [B, 2]

        # 将 raw（可能已 tanh）映射到角度范围
        if not self.use_tanh_head:
            raw = torch.tanh(raw)
        pitch = 90.0 * raw[:, 0]
        roll  = 180.0 * raw[:, 1]
        angles = torch.stack([pitch, roll], dim=1)
        return angles

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalContrast(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        # 多模态投影头
        self.proj_atomic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_residue = nn.Sequential(
            nn.Linear(150, 256),  # 残基特征维度150
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.temperature = temperature

    def forward(self, atomic_repr, residue_repr, labels):
        """
        atomic_repr: [batch, n_atom, hidden_dim]
        residue_repr: [batch, n_res, 150]
        coord_repr: [batch, n_atom, 3]
        labels: 用于构造正负对的分子标识
        """
        # 模态特征投影
        h_atomic = F.normalize(self.proj_atomic(atomic_repr.mean(1)), dim=-1) # [B,128]
        h_residue = F.normalize(self.proj_residue(residue_repr.mean(1)), dim=-1)
        
        # 构造跨模态对比
        loss = 0
        # 原子-残基对比
        loss += self.cross_modal_contrast(h_atomic, h_residue, labels)
        
        return loss #/ 3  # 平均三种对比损失

    def cross_modal_contrast(self, mod1, mod2, labels):

        labels = labels.squeeze() 
        # 计算模态间相似度矩阵
        sim_matrix = mod1 @ mod2.T / self.temperature  # [B,B]
        
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        # 调试语句（建议正式运行时注释掉）
        print("相似度矩阵尺寸:", sim_matrix.size())  # 应为[N, M]
        print("掩码矩阵尺寸:", pos_mask.size())     # 应与sim_matrix一致
        
        # 计算对比损失
        positives = sim_matrix[pos_mask].view(-1,1)
        negatives = sim_matrix[~pos_mask]#.view(sim_matrix.size(0), -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        return F.cross_entropy(logits, labels)

class RelationalContrast(nn.Module):
    def __init__(self, hidden_dim=1174, temperature=0.1):
        #(CLS：1)/1024+150
        super().__init__()
        # 关系编码器
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.temperature = temperature
        self.relation_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.LayerNorm(32)
        ])
        
        # 将各层移动到指定设备
        for layer in self.relation_layers:
            layer.to(self.device)      

    def _forward_relation(self, x):
        """安全的逐层前向传播"""
        for layer in self.relation_layers:
            # 仅参数层需要显式设备对齐
            if hasattr(layer, 'weight'):
                x = x.to(layer.weight.device)
            x = layer(x)
        return x
    
    def forward(self, atom_feat, res_feat, adjacency):
        """
        atom_feat: [B, N_atom, D_atom]
        res_feat: [B, N_res, D_res]
        adjacency: [B, N_atom, N_res]
        """

        atom_feat = atom_feat.to(self.device)
        res_feat = res_feat.to(self.device)
        adjacency = adjacency.to(self.device)
        res_expanded = torch.einsum('brd,bnr->bnd', res_feat, adjacency.float())
        # 构建关系对
        pairs = torch.cat([atom_feat, res_expanded], dim=-1)  # [B, N_atom, D_atom+D_res]
        relations = self._forward_relation(pairs)
        B,N,_ = relations.shape
        # 归一化
        relations = F.normalize(relations, p=2, dim=-1)
        relations = relations.unsqueeze(2).expand(-1, -1, N, -1)  # [B,128,128,32]
        # 生成原子到残基的邻接矩阵（对角矩阵变体）
        adjacency = torch.eye(N, N)  # [128,128] 对角矩阵
        adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)  # [B,128,128]

        # 正样本：原子与所属残基的对应位置
        pos_mask = adjacency > 0.5  # [B,128,128] 对角线为True

        # 负样本：随机选择非对角元素
        neg_mask = torch.rand_like(adjacency.float()) > 0.9  # 随机采样10%负样本
        neg_mask = neg_mask & (~pos_mask)  # 排除正样本位置

        #print(relations.shape,pos_mask.shape,adjacency.shape)
        # 正样本特征提取
        pos_relations = relations[pos_mask]  # [B*128,32] → 每个原子取对应残基的特征

        # 负样本特征采样
        neg_relations = relations[neg_mask]  # [B*N_neg,32]

        # 计算对比损失
        logits = torch.mm(pos_relations, neg_relations.T).to(self.device)  # [N_pos, N_neg]
        labels = torch.arange(pos_relations.size(0)).to(self.device)  # 确保labels在GPU
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss
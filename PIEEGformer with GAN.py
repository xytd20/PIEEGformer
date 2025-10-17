#---------------------------------------
# 改进的动态Dropout
#---------------------------------------
class DynamicDropout(nn.Module):
    """
    动态调整的Dropout层
    在训练初期使用较低的dropout率，随着训练进行逐渐增加
    """
    def __init__(self, p_min=0.3, p_max=0.7, annealing_epochs=40):
        super(DynamicDropout, self).__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.annealing_epochs = annealing_epochs
        self.current_p = p_min
        self.epoch = 0
        self.training_mode = True
        
    def forward(self, x):
        if self.training and self.training_mode:
            p = self.current_p
            return F.dropout(x, p=p, training=True)
        else:
            return x
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        if epoch >= self.annealing_epochs:
            self.current_p = self.p_max
        else:
            # 线性增加dropout率
            self.current_p = self.p_min + (self.p_max - self.p_min) * (epoch / self.annealing_epochs)
            
    def get_current_p(self):
        return self.current_p
    
    def set_training_mode(self, mode):
        self.training_mode = mode

#---------------------------------------
# 损失函数定义
#---------------------------------------
class FocalLoss(nn.Module):
    """针对不平衡类别的Focal Loss"""
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # 直接存储alpha值而不是注册buffer
            
    def forward(self, input, target):
        # 确保输入和目标在同一设备上
        device = input.device
        
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            # 确保alpha在正确的设备上
            alpha = torch.tensor(self.alpha, dtype=torch.float32, device=device)
            alpha_t = alpha[target]
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

#---------------------------------------
# 空间位置编码模块
#---------------------------------------
class SpatialPositionEncoding(nn.Module):
    """用于编码EEG电极的空间位置信息"""
    def __init__(self, n_channels=30, embedding_dim=32):
        super(SpatialPositionEncoding, self).__init__()
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        
        # 存储电极位置信息 - 预定义的三维坐标 (x, y, z)
        self.electrode_positions = torch.tensor([
            [0.707106781186548, 0, 0.707106781186548],               # Fz
            [0.673028145070219, 0.545007445768716, 0.500000000000000],               # F3
            [0.349725383249891, 0.865600698387781, 0.358367949545300],               # FC5
            [0.370487385972602, 0.357775509841357, 0.857167300702112],               # FC1
            [4.32978028117747e-17, 0.707106781186548, 0.707106781186548],            # C3
            [-0.334565303179429, 0.871572412738697, 0.358367949545300],              # CP5
            [-0.370487385972602, 0.357775509841357, 0.857167300702112],              # CP1
            [-0.707106781186548, -8.65956056235493e-17, 0.707106781186548],          # Pz
            [-0.334565303179429, -0.871572412738697, 0.358367949545300],             # CP6
            [-0.370487385972602, -0.357775509841357, 0.857167300702112],             # CP2
            [6.12323399573677e-17, 0, 1],                                            # Cz
            [4.32978028117747e-17, -0.707106781186548, 0.707106781186548],           # C4
            [0.334565303179429, -0.871572412738697, 0.358367949545300],              # FC6
            [0.370487385972602, -0.357775509841357, 0.857167300702112],              # FC2
            [0.673028145070219, -0.545007445768716, 0.500000000000000],              # F4
            [0.699754537669432, 0.282719184865606, 0.656059028990507],               # F1
            [0.365890464984075, 0.660083872029737, 0.656059028990507],               # FC3
            [2.39253812915811e-17, 0.390731128489274, 0.920504853452440],            # C1
            [5.67736369858161e-17, 0.927183854566787, 0.374606593415912],            # C5
            [-0.365890464984075, 0.660083872029737, 0.656059028990507],              # CP3
            [-0.699754537669432, 0.282719184865606, 0.656059028990507],              # P1
            [-0.920504853452440, -1.12729332238013e-16, 0.390731128489274],          # POz
            [-0.699754537669432, -0.282719184865606, 0.656059028990507],             # P2
            [-0.374606593415912, -4.58760765566291e-17, 0.927183854566787],          # CPz
            [-0.365890464984075, -0.660083872029737, 0.656059028990507],             # CP4
            [5.67736369858161e-17, -0.927183854566787, 0.374606593415912],           # C6
            [2.39253812915811e-17, -0.390731128489274, 0.920504853452440],           # C2
            [0.365890464984075, -0.660083872029737, 0.656059028990507],              # FC4
            [0.699754537669432, -0.282719184865606, 0.656059028990507],              # F2
            [0.390731128489274, 0, 0.920504853452440]                                # FCz
        ], dtype=torch.float32)
        
        # 改进的投影层 - 增加泛化能力
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            DynamicDropout(p_min=0.05, p_max=0.1, annealing_epochs=30)  # 使用DynamicDropout
        )
        
        # 添加空间相关性编码 - 计算电极间的相对距离关系
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习的融合权重
    
    def update_dropout(self, epoch):
        """更新动态dropout率"""
        for module in self.modules():
            if isinstance(module, DynamicDropout):
                module.update_epoch(epoch)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        batch_size, seq_len, channels = x.size()
        
        # 编码电极绝对位置
        positions = self.electrode_positions.to(x.device)  # [channels, 3]
        position_embeddings = self.position_encoder(positions)  # [channels, embedding_dim]
        
        # 计算电极间相对距离矩阵
        dist_matrix = torch.cdist(positions, positions, p=2)  # [channels, channels]
        dist_matrix = dist_matrix.unsqueeze(-1)  # [channels, channels, 1]
        
        # 编码电极间的相对距离
        distance_embeddings = self.distance_encoder(dist_matrix)  # [channels, channels, embedding_dim//4]
        
        # 聚合距离编码
        distance_features = distance_embeddings.mean(dim=1)  # [channels, embedding_dim//4]
        
        # 扩展维度，使其与position_embeddings兼容
        distance_features = distance_features.repeat(1, 4)  # [channels, embedding_dim]
        
        # 融合绝对位置和相对距离编码
        combined_embeddings = position_embeddings + self.alpha * distance_features
        
        # 创建空间嵌入张量并扩展到所需维度
        # [batch_size, seq_len, channels, embedding_dim]
        return combined_embeddings.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

#---------------------------------------
# 运动模式特征提取模块
#---------------------------------------
class MotorPatternExtractor(nn.Module):
    """专门提取运动相关脑电特征的模块"""
    def __init__(self, n_channels=30, seq_len=51, n_filters=32):
        super(MotorPatternExtractor, self).__init__()
        self.n_channels = n_channels
        
        # μ节律特征提取 (8-12 Hz)
        self.mu_extractor = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=9, padding=4),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
            nn.MaxPool1d(2, stride=1, padding=1)
        )
        
        # β节律特征提取 (13-30 Hz)
        self.beta_extractor = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
            nn.MaxPool1d(2, stride=1, padding=1)
        )
        
        # 低频MRCP特征提取 (0.1-5 Hz)
        self.mrcp_extractor = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=13, padding=6),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
            nn.MaxPool1d(2, stride=1, padding=1)
        )
        
        # 运动区域注意力 (针对运动皮层区域的注意力机制)
        self.motor_attention = nn.Sequential(
            nn.Conv1d(n_filters*3, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(n_filters*3, n_filters*2, kernel_size=1),
            nn.BatchNorm1d(n_filters*2),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, channels]
        # 转换为卷积格式
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 提取不同频带特征
        mu_features = self.mu_extractor(x)
        beta_features = self.beta_extractor(x)
        mrcp_features = self.mrcp_extractor(x)
        
        # 合并特征
        combined = torch.cat([mu_features, beta_features, mrcp_features], dim=1)
        
        # 运动区域注意力
        attention = self.motor_attention(combined)
        
        # 加权融合
        attended_features = combined * attention
        
        # 特征降维
        output = self.fusion(attended_features)
        
        # 返回到序列格式
        output = output.permute(0, 2, 1)  # [batch_size, seq_len, filters*2]
        
        return output

#---------------------------------------
# 运动检测专用的注意力机制
#---------------------------------------
class MovementDetectionAttention(nn.Module):
    """改进的运动检测专用注意力机制，整合时域和空间位置信息"""
    def __init__(self, input_dim, spatial_dim=32, hidden_dim=64):
        super(MovementDetectionAttention, self).__init__()
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        
        # 时间注意力层 - 关注运动准备关键时刻
        self.temporal_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 空间注意力层 - 关注运动相关脑区
        self.spatial_attention = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 运动模式注意力 - 识别特定运动特征模式
        self.pattern_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 门控机制 - 控制信息流
        self.gate = nn.Sequential(
            nn.Linear(input_dim + spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, spatial_embeddings=None):
        # x shape: [batch_size, seq_len, channels]
        batch_size, seq_len, channels = x.shape
        
        # 时间注意力 - 找出关键时间点
        temporal_weights = self.temporal_attention(x)  # [batch_size, seq_len, 1]
        
        # 运动模式注意力 - 识别特定模式
        pattern_weights = self.pattern_attention(x)
        pattern_enhanced = x * pattern_weights
        
        # 空间注意力 - 如果提供了空间嵌入
        if spatial_embeddings is not None:
            try:
                # 对通道维度取平均，得到 [batch_size, seq_len, spatial_dim]
                spatial_input = spatial_embeddings.mean(dim=2)
                
                # 计算空间注意力权重 [batch_size, seq_len, 1]
                spatial_attn = self.spatial_attention(spatial_input)
                
                # 应用空间注意力
                x_spatial = pattern_enhanced * spatial_attn
                
                # 应用门控机制 - 融合普通特征和空间增强特征
                combined_input = torch.cat([pattern_enhanced, spatial_input], dim=-1)
                gate_value = self.gate(combined_input)
                
                enhanced_features = gate_value * x_spatial + (1 - gate_value) * pattern_enhanced
            except Exception as e:
                print(f"空间注意力处理警告: {e}")
                enhanced_features = pattern_enhanced
        else:
            enhanced_features = pattern_enhanced
        
        # 加权平均 - 时间维度
        attended_features = torch.sum(enhanced_features * temporal_weights, dim=1)  # [batch_size, channels]
        
        # 残差连接 - 保留原始信息
        final_features = attended_features + x.mean(dim=1)
        
        return final_features, temporal_weights

#---------------------------------------
# 残差块 - 提高深层网络训练稳定性
#---------------------------------------
class ResidualBlock(nn.Module):
    """带有残差连接的基本块，提高梯度流动和训练稳定性"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.1, dropout_p_max=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = DynamicDropout(p_min=dropout_p, p_max=dropout_p_max, annealing_epochs=30)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = DynamicDropout(p_min=dropout_p, p_max=dropout_p_max, annealing_epochs=30)
        
        # 如果输入输出通道数不同，使用1x1卷积进行映射
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def update_dropout(self, epoch):
        """更新动态dropout率"""
        self.dropout1.update_epoch(epoch)
        self.dropout2.update_epoch(epoch)
            
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        out = self.dropout2(out)
        return out
    
class EnhancedMergeNetwork(nn.Module):
    """合并网络，处理统一的Transformer特征维度"""
    def __init__(self, feature_dim=256):
        super(EnhancedMergeNetwork, self).__init__()
        self.feature_dim = feature_dim
        
        print(f"初始化增强型合并网络，特征维度: {feature_dim}")
        
        # 特征转换
        self.mrcp_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        
        self.erd_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        
        # 注意力融合
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合后的特征处理
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            DynamicDropout(p_min=0.2, p_max=0.5, annealing_epochs=40),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            DynamicDropout(p_min=0.2, p_max=0.5, annealing_epochs=40),
            nn.Linear(64, 2)
        )
        
        # 残差分类器
        self.residual_classifier = nn.Linear(feature_dim * 2, 2)
    
    def update_dropout(self, epoch):
        """更新动态dropout率"""
        for module in self.modules():
            if isinstance(module, DynamicDropout):
                module.update_epoch(epoch)
    
    def forward(self, mrcp_features, erd_features):
        # 假设输入已经是统一的2D特征 [batch, features]
        assert mrcp_features.dim() == 2 and erd_features.dim() == 2, "特征维度必须是2D"
        assert mrcp_features.size(1) == self.feature_dim and erd_features.size(1) == self.feature_dim, \
            f"特征维度不匹配，期望{self.feature_dim}，获得mrcp:{mrcp_features.size(1)}, erd:{erd_features.size(1)}"
        
        # 转换特征
        enhanced_mrcp = self.mrcp_transform(mrcp_features)
        enhanced_erd = self.erd_transform(erd_features)
        
        # 计算注意力权重
        combined = torch.cat([enhanced_mrcp, enhanced_erd], dim=1)
        weights = self.attention(combined)
        
        # 加权融合
        weighted_mrcp = enhanced_mrcp * weights[:, 0].unsqueeze(1)
        weighted_erd = enhanced_erd * weights[:, 1].unsqueeze(1)
        
        # 求和融合
        fused_features = weighted_mrcp + weighted_erd
        
        # 主分类器
        main_logits = self.fusion_network(fused_features)
        
        # 残差分类器
        res_logits = self.residual_classifier(combined)
        
        # 线性组合两个分类器的输出
        alpha = 0.9  # 主分类器权重
        logits = alpha * main_logits + (1 - alpha) * res_logits
        
        return logits

#---------------------------------------
# Transformer增强版EEG分类器
#---------------------------------------
class EnhancedTransformerEEGClassifier(nn.Module):
    """使用Transformer架构的增强EEG分类器，支持任意长度的时间序列，兼容原始调用方式"""
    def __init__(self, seq_len=51, n_channels=30, n_classes=2, spatial_dim=32, nhead=8, num_layers=6):
        super(EnhancedTransformerEEGClassifier, self).__init__()
        self.seq_len = seq_len  # 保留这个参数以保持兼容性
        self.n_channels = n_channels
        self.feature_dim = 256
        self.max_seq_len = 500  # 支持的最大序列长度
        
        print(f"初始化增强Transformer模型，支持长序列输入 (最大长度: {self.max_seq_len})")
        
        # 空间位置编码
        self.spatial_encoder = SpatialPositionEncoding(n_channels, spatial_dim)
        
        # 运动模式特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # 位置编码层 - 使用可学习的相对位置编码
        self.pos_embedding = nn.Embedding(self.max_seq_len, 64)
        
        # Transformer编码器 - 使用动态dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,  # 这里保持固定dropout，因为TransformerEncoderLayer内部管理
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 自适应池化 - 支持任意长度序列
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        # 全局特征处理 - 使用DynamicDropout
        self.global_pool = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            DynamicDropout(p_min=0.1, p_max=0.5, annealing_epochs=40)
        )
        
        # 分类头 - 使用DynamicDropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            DynamicDropout(p_min=0.2, p_max=0.5, annealing_epochs=40),
            nn.Linear(128, n_classes)
        )
        
    def update_dropout(self, epoch):
        """更新所有动态dropout层的率"""
        # 更新global_pool中的dropout
        for module in self.global_pool.modules():
            if isinstance(module, DynamicDropout):
                module.update_epoch(epoch)
        
        # 更新classifier中的dropout
        for module in self.classifier.modules():
            if isinstance(module, DynamicDropout):
                module.update_epoch(epoch)
                
        # 更新spatial_encoder中的dropout
        if hasattr(self.spatial_encoder, 'update_dropout'):
            self.spatial_encoder.update_dropout(epoch)
        
    def forward(self, x):
        # 输入 x: [batch_size, seq_len, channels]
        batch_size, seq_len, channels = x.size()
        
        # 安全检查
        if seq_len > self.max_seq_len:
            print(f"警告: 输入序列长度 {seq_len} 超过最大长度 {self.max_seq_len}，将被截断")
            x = x[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
        
        # 调整为卷积格式
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 特征提取
        x = self.feature_extractor(x)  # [batch_size, 64, seq_len]
        
        # 调整回序列格式
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 64]
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, max=self.max_seq_len-1)  # 确保不超过嵌入表大小
        position_embeddings = self.pos_embedding(position_ids)
        
        x = x + position_embeddings
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, seq_len, 64]
        
        # 自适应池化 - 处理可变长度序列
        x = x.transpose(1, 2)  # [batch_size, 64, seq_len]
        x = self.adaptive_pool(x).squeeze(-1)  # [batch_size, 64]
        
        # 全局特征处理
        x = self.global_pool(x)  # [batch_size, 256]
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def extract_features(self, x):
        """提取中间特征用于迁移学习或特征可视化"""
        batch_size, seq_len, channels = x.size()
        
        # 安全检查
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
        
        # 调整为卷积格式
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 特征提取
        features = self.feature_extractor(x)  # [batch_size, 64, seq_len]
        
        # 调整回序列格式
        x = features.permute(0, 2, 1)  # [batch_size, seq_len, 64]
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, max=self.max_seq_len-1)
        position_embeddings = self.pos_embedding(position_ids)
        
        x = x + position_embeddings
        
        # Transformer编码
        transformer_output = self.transformer_encoder(x)
        
        # 自适应池化
        pooled = transformer_output.transpose(1, 2)  # [batch_size, 64, seq_len]
        pooled = self.adaptive_pool(pooled).squeeze(-1)  # [batch_size, 64]
        
        # 全局特征处理
        global_features = self.global_pool(pooled)
        
        return global_features, transformer_output, features.permute(0, 2, 1)
    
#---------------------------------------
# 改进的GAN模型定义
#---------------------------------------
class EnhancedEEGGenerator(nn.Module):
    """改进的EEG数据生成器，专注于生成高质量运动相关模式"""
    def __init__(self, latent_dim=100, seq_len=51, channels=30, n_classes=2):
        super(EnhancedEEGGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.channels = channels
        self.n_classes = n_classes
        
        # 类别条件编码 - 增强条件控制
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, 32),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2)
        )
        
        # 初始全连接映射
        self.initial_dense = nn.Sequential(
            nn.Linear(latent_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, seq_len * 64),
            nn.BatchNorm1d(seq_len * 64),
            nn.LeakyReLU(0.2)
        )
        
        # 残差上采样块 - 使用DynamicDropout
        self.res_upsample1 = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128, 128, dropout_p=0.05, dropout_p_max=0.1)
        )
        
        self.res_upsample2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64, 64, dropout_p=0.05, dropout_p_max=0.1)
        )
        
        # 频率特性融合层 - 生成不同频率的EEG特征
        self.freq_conv_low = nn.Conv1d(64, 32, kernel_size=9, padding=4)  # 低频特征
        self.freq_conv_mid = nn.Conv1d(64, 32, kernel_size=5, padding=2)  # 中频特征
        self.freq_conv_high = nn.Conv1d(64, 32, kernel_size=3, padding=1)  # 高频特征
        
        # 最终通道投影
        self.final_conv = nn.Conv1d(96, channels, kernel_size=1)
        
        # EEG信号特性修正层 - 添加EEG特有的连续性和平滑特性
        self.smoothing = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.Tanh()
        )
    
    def update_dropout(self, epoch):
        """更新动态dropout率"""
        # 更新残差块中的dropout
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                module.update_dropout(epoch)
        
    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # 嵌入标签和条件控制
        label_embedding = self.label_embedding(labels)
        
        # 合并噪声和标签信息
        combined = torch.cat([z, label_embedding], dim=1)
        
        # 初始全连接映射
        x = self.initial_dense(combined)
        x = x.view(batch_size, 64, self.seq_len)
        
        # 残差上采样
        x = self.res_upsample1(x)
        x = self.res_upsample2(x)
        
        # 调整到目标序列长度
        x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        # 多频率特征生成
        low_freq = self.freq_conv_low(x)
        mid_freq = self.freq_conv_mid(x)
        high_freq = self.freq_conv_high(x)
        
        # 合并不同频率特征
        multi_freq = torch.cat([low_freq, mid_freq, high_freq], dim=1)
        
        # 生成通道数据
        eeg_data = self.final_conv(multi_freq)
        
        # 信号平滑处理
        eeg_data = self.smoothing(eeg_data)
        
        # 调整输出格式
        eeg_data = eeg_data.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        return eeg_data

class EnhancedEEGDiscriminator(nn.Module):
    """改进的EEG鉴别器，专注于识别真实的运动模式"""
    def __init__(self, seq_len=51, channels=30, n_classes=2):
        super(EnhancedEEGDiscriminator, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.n_classes = n_classes
        
        # 类别嵌入
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, 32),
            nn.Linear(32, channels),
            nn.LeakyReLU(0.2)
        )
        
        # 特征提取层 - 使用运动模式提取器
        self.pattern_extractor = MotorPatternExtractor(channels, seq_len, n_filters=16)
        
        # 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32)
            ),
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32)
            ),
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=7, padding=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32)
            )
        ])
        
        # 深度残差处理 - 使用DynamicDropout
        self.res_layers = nn.Sequential(
            ResidualBlock(96, 128, dropout_p=0.05, dropout_p_max=0.2),
            nn.AvgPool1d(2),
            ResidualBlock(128, 128, dropout_p=0.05, dropout_p_max=0.2),
            nn.AvgPool1d(2),
            ResidualBlock(128, 256, dropout_p=0.05, dropout_p_max=0.2)
        )
        
        # 计算展平后的维度
        self._get_flat_dim = self._compute_flat_dim()
        
        # 条件判别器 - 使用DynamicDropout
        self.conditional_disc = nn.Sequential(
            nn.Linear(self._get_flat_dim + channels, 512),
            nn.LeakyReLU(0.2),
            DynamicDropout(p_min=0.2, p_max=0.5, annealing_epochs=40),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            DynamicDropout(p_min=0.2, p_max=0.5, annealing_epochs=40),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def update_dropout(self, epoch):
        """更新动态dropout率"""
        # 更新残差层中的dropout
        for module in self.res_layers.modules():
            if isinstance(module, ResidualBlock):
                module.update_dropout(epoch)
        
        # 更新条件判别器中的dropout
        for module in self.conditional_disc.modules():
            if isinstance(module, DynamicDropout):
                module.update_epoch(epoch)
        
    def _compute_flat_dim(self):
        """计算卷积层输出的扁平化维度"""
        with torch.no_grad():
            x = torch.zeros(1, self.channels, self.seq_len)
            pattern_out = self.pattern_extractor(x.permute(0, 2, 1)).permute(0, 2, 1)
            
            multi_scale_out = []
            for conv in self.multi_scale_conv:
                multi_scale_out.append(conv(pattern_out))
            
            combined = torch.cat(multi_scale_out, dim=1)
            res_out = self.res_layers(combined)
            
            return res_out.flatten().size(0)
    
    def forward(self, x, labels):
        # x shape: [batch_size, seq_len, channels]
        batch_size = x.size(0)
        
        # 提取标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 提取运动模式特征
        pattern_features = self.pattern_extractor(x)
        
        # 转换为卷积格式
        pattern_features = pattern_features.permute(0, 2, 1)  # [batch_size, 64, seq_len]
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(pattern_features))
        
        # 合并特征
        combined = torch.cat(multi_scale_features, dim=1)  # [batch_size, 96, seq_len]
        
        # 应用残差层
        features = self.res_layers(combined)
        
        # 展平
        flat_features = features.view(batch_size, -1)
        
        # 合并标签嵌入
        conditional_input = torch.cat([flat_features, label_emb], dim=1)
        
        # 判别结果
        validity = self.conditional_disc(conditional_input)
        
        return validity

#---------------------------------------
# 数据加载器类 - 支持合并时间窗口
#---------------------------------------
class EEGDataLoader:
    """
    数据加载器，用于EEG数据，支持多个时间窗口的合并
    """
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.subjects = [5, 6, 7, 8, 11, 12, 13, 16, 18, 20]
        self.time_windows = ['-1s~-0.8s', '-0.9s~-0.7s', '-0.8s~-0.6s', 
                         '-0.7s~-0.5s', '-0.6s~-0.4s', '-0.5s~-0.3s', 
                         '-0.4s~-0.2s', '-0.3s~-0.1s', '-0.2s~0s']
        self.data_types = ['MRCP', 'ERD']
        self.raw_data = {}
        self.processed_data = {}
        
        # 定义两个二分类任务
        self.tasks = {
            'left_hand': {'name': 'Left Hand Move/NoMove', 'classes': [0, 2]},  # Go-Left vs NoGo-Left
            'right_hand': {'name': 'Right Hand Move/NoMove', 'classes': [1, 3]}  # Go-Right vs NoGo-Right
        }
        
        # 添加合并窗口
        self.merged_windows = {
            '-0.5s~-0.1s': ['-0.5s~-0.3s', '-0.3s~-0.1s'],  # 合并-0.5到-0.1秒的时间窗口
        }
    
    def load_data(self):
        print("为所有受试者加载数据...")
        for subject_id in self.subjects:
            self.raw_data[subject_id] = {}
            for data_type in self.data_types:
                filename = f"sub{subject_id}_Data_window_{data_type}.mat"
                file_path = os.path.join(self.data_dir, filename)
                if not os.path.exists(file_path):
                    print(f"警告: 文件未找到: {file_path}")
                    continue
                try:
                    self.raw_data[subject_id][data_type] = sio.loadmat(file_path)
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {str(e)}")
        print("数据加载完成。")
    
    def prepare_data_for_classification(self, task='left_hand'):
        """准备分类数据，针对特定的二分类任务"""
        print(f"准备分类数据，任务: {self.tasks[task]['name']}")
        
        # 初始化数据存储 - 标准时间窗口
        for time_window_str_idx, time_window_str in enumerate(self.time_windows):
            self.processed_data[time_window_str] = {
                'eeg_mrcp_data': [],
                'eeg_erd_data': [],
                'labels': [],
                'subjects': [],
                'trial_ids': []
            }
        
        # 初始化合并时间窗口
        for merged_window, _ in self.merged_windows.items():
            self.processed_data[merged_window] = {
                'eeg_mrcp_data': [],
                'eeg_erd_data': [],
                'labels': [],
                'subjects': [],
                'trial_ids': []
            }
        
        # 获取当前任务的类别映射
        task_classes = self.tasks[task]['classes']
        
        # 处理每个受试者的数据
        for subject_id in self.subjects:
            if subject_id not in self.raw_data:
                continue
            subject_raw_mat_data = self.raw_data[subject_id]

            if 'MRCP' not in subject_raw_mat_data or 'ERD' not in subject_raw_mat_data:
                print(f"警告: 受试者 {subject_id} 缺少MRCP或ERD数据。跳过。")
                continue
            
            mrcp_mat_content = subject_raw_mat_data['MRCP']
            erd_mat_content = subject_raw_mat_data['ERD']
            
            # 原始的四分类映射
            condition_map = {
                ('go_EL', 'go_ML'): 0,  # 左手动作-Go
                ('go_ER', 'go_MR'): 1,  # 右手动作-Go
                ('nogo_EL', 'nogo_ML'): 2,  # 左手不动-NoGo
                ('nogo_ER', 'nogo_MR'): 3   # 右手不动-NoGo
            }
            
            # 处理标准时间窗口数据
            for (eeg_field, _), original_label in condition_map.items():
                # 只处理当前任务相关的类别
                if original_label not in task_classes:
                    continue
                    
                # 将原始标签映射到二分类标签 (0或1)
                binary_label = task_classes.index(original_label)
                
                if not (eeg_field in mrcp_mat_content and eeg_field in erd_mat_content):
                    continue
                    
                mrcp_cells_for_condition = mrcp_mat_content[eeg_field]
                erd_cells_for_condition = erd_mat_content[eeg_field]

                if not (isinstance(mrcp_cells_for_condition, np.ndarray) and mrcp_cells_for_condition.shape == (1,9) and
                        isinstance(erd_cells_for_condition, np.ndarray) and erd_cells_for_condition.shape == (1,9)):
                    continue

                # 处理每个标准时间窗口
                for window_idx, time_window_str_val in enumerate(self.time_windows):
                    mrcp_data_in_cell = mrcp_cells_for_condition[0, window_idx]
                    erd_data_in_cell = erd_cells_for_condition[0, window_idx]

                    if not (isinstance(mrcp_data_in_cell, np.ndarray) and isinstance(erd_data_in_cell, np.ndarray)):
                        continue
                    
                    if mrcp_data_in_cell.size == 0 or erd_data_in_cell.size == 0:
                        continue
                    
                    if not (mrcp_data_in_cell.ndim == 3 and erd_data_in_cell.ndim == 3):
                        continue

                    if not (mrcp_data_in_cell.shape[0] == 51 and mrcp_data_in_cell.shape[1] == 30 and
                            erd_data_in_cell.shape[0] == 51 and erd_data_in_cell.shape[1] == 30):
                        continue
                    
                    num_trials = min(mrcp_data_in_cell.shape[2], erd_data_in_cell.shape[2])
                    if num_trials == 0:
                        continue

                    for trial_i in range(num_trials):
                        mrcp_single_trial = mrcp_data_in_cell[:, :, trial_i]
                        erd_single_trial = erd_data_in_cell[:, :, trial_i]
                        
                        self.processed_data[time_window_str_val]['eeg_mrcp_data'].append(mrcp_single_trial)
                        self.processed_data[time_window_str_val]['eeg_erd_data'].append(erd_single_trial)
                        self.processed_data[time_window_str_val]['labels'].append(binary_label)  # 使用二分类标签
                        self.processed_data[time_window_str_val]['subjects'].append(subject_id)
                        
                        trial_id = f"{subject_id}_{trial_i}_{eeg_field}_{time_window_str_val}"
                        self.processed_data[time_window_str_val]['trial_ids'].append(trial_id)
                        
                        # 记录窗口索引，用于后续合并窗口查找
                        self.processed_data[time_window_str_val]['window_index'] = window_idx
        
        # 转换标准时间窗口数据为numpy数组
        for time_window_key in self.time_windows:
            if not self.processed_data[time_window_key]['labels']:
                print(f"警告: 时间窗口 {time_window_key} 没有加载数据。")
                self.processed_data[time_window_key]['labels'] = np.array([])
                self.processed_data[time_window_key]['subjects'] = np.array([])
                self.processed_data[time_window_key]['trial_ids'] = np.array([])
                continue

            # 转换为numpy数组
            self.processed_data[time_window_key]['eeg_mrcp_data'] = np.array(self.processed_data[time_window_key]['eeg_mrcp_data'])
            self.processed_data[time_window_key]['eeg_erd_data'] = np.array(self.processed_data[time_window_key]['eeg_erd_data'])
            self.processed_data[time_window_key]['labels'] = np.array(self.processed_data[time_window_key]['labels'])
            self.processed_data[time_window_key]['subjects'] = np.array(self.processed_data[time_window_key]['subjects'])
            self.processed_data[time_window_key]['trial_ids'] = np.array(self.processed_data[time_window_key]['trial_ids'])
            
            num_total_samples = len(self.processed_data[time_window_key]['labels'])
            class_counts_arr = np.bincount(self.processed_data[time_window_key]['labels'], minlength=2)
            
            # 显示二分类的类别名称
            class_names = ['Move', 'NoMove'] if task == 'left_hand' else ['Move', 'NoMove']
            print(f"时间窗口 {time_window_key}: {num_total_samples} 个样本。")
            for i, count in enumerate(class_counts_arr):
                print(f"  {class_names[i]}: {count} 个样本")
        
        # 处理合并时间窗口
        self._create_merged_windows()
        
        print("数据准备完成，包括标准窗口和合并窗口。")
    
    def _create_merged_windows(self):
        """
        创建合并的时间窗口数据 - 处理重叠边界
        """
        print("正在创建拼接时间窗口...")
        
        for merged_window, component_windows in self.merged_windows.items():
            print(f"处理拼接窗口: {merged_window} (组合: {', '.join(component_windows)})")
            merged_data = self.processed_data[merged_window]
            
            # 收集所有组件窗口的试验ID以识别共同试验
            all_trial_bases = set()
            for window in component_windows:
                if window not in self.processed_data:
                    print(f"  跳过不存在的窗口: {window}")
                    continue
                    
                window_data = self.processed_data[window]
                if len(window_data['trial_ids']) == 0:
                    print(f"  窗口 {window} 没有数据，跳过")
                    continue
                    
                # 提取试验基础ID (去掉时间窗口信息)
                trial_bases = [id.rsplit('_', 1)[0] for id in window_data['trial_ids']]
                all_trial_bases.update(trial_bases)
            
            print(f"  找到 {len(all_trial_bases)} 个独立试验基础")
            
            # 对每个基础试验，拼接不同时间窗口的数据
            for trial_base in all_trial_bases:
                # 检查该试验是否在所有组件窗口中都存在
                all_windows_present = True
                trial_indices = {}
                
                for window in component_windows:
                    if window not in self.processed_data:
                        all_windows_present = False
                        break
                        
                    window_data = self.processed_data[window]
                    if len(window_data['trial_ids']) == 0:
                        all_windows_present = False
                        break
                    
                    # 查找匹配的试验
                    trial_bases_in_window = [id.rsplit('_', 1)[0] for id in window_data['trial_ids']]
                    indices = [i for i, base in enumerate(trial_bases_in_window) if base == trial_base]
                    
                    if not indices:
                        all_windows_present = False
                        break
                    
                    # 保存该窗口中该试验的索引
                    trial_indices[window] = indices[0]
                
                # 如果某些窗口缺少该试验，则跳过
                if not all_windows_present:
                    continue
                
                # 确保所有窗口的标签一致
                trial_labels = []
                trial_subjects = []
                
                for window in component_windows:
                    idx = trial_indices[window]
                    trial_labels.append(self.processed_data[window]['labels'][idx])
                    trial_subjects.append(self.processed_data[window]['subjects'][idx])
                
                if len(set(trial_labels)) > 1:
                    print(f"  跳过标签不一致的试验: {trial_base}")
                    continue
                
                try:
                    # 沿时间维度拼接数据，处理重叠
                    mrcp_concat_parts = []
                    erd_concat_parts = []
                    
                    # 按顺序拼接，注意处理重叠
                    for i, window in enumerate(component_windows):
                        idx = trial_indices[window]
                        window_data = self.processed_data[window]
                        
                        # 提取当前窗口数据
                        mrcp_data = window_data['eeg_mrcp_data'][idx]  # (51, 30)
                        erd_data = window_data['eeg_erd_data'][idx]    # (51, 30)
                        
                        # 如果不是第一个窗口，需要去掉第一个时间点以避免重叠
                        if i > 0:
                            mrcp_data = mrcp_data[1:, :]  # 去掉第一个时间点
                            erd_data = erd_data[1:, :]
                        
                        mrcp_concat_parts.append(mrcp_data)
                        erd_concat_parts.append(erd_data)
                    
                    # 沿时间维度（第0维）拼接
                    mrcp_concatenated = np.concatenate(mrcp_concat_parts, axis=0)  # (101, 30)
                    erd_concatenated = np.concatenate(erd_concat_parts, axis=0)    # (101, 30)
                    
                    print(f"  拼接窗口 {merged_window} 的形状: {mrcp_concatenated.shape} (应为101x30)")
                    
                    # 添加到合并窗口数据
                    merged_data['eeg_mrcp_data'].append(mrcp_concatenated)
                    merged_data['eeg_erd_data'].append(erd_concatenated)
                    merged_data['labels'].append(trial_labels[0])
                    merged_data['subjects'].append(trial_subjects[0])
                    merged_data['trial_ids'].append(f"{trial_base}_{merged_window}")
                    
                except Exception as e:
                    print(f"  拼接窗口数据时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 转换为numpy数组
            if merged_data['labels']:
                merged_data['eeg_mrcp_data'] = np.array(merged_data['eeg_mrcp_data'])
                merged_data['eeg_erd_data'] = np.array(merged_data['eeg_erd_data'])
                merged_data['labels'] = np.array(merged_data['labels'])
                merged_data['subjects'] = np.array(merged_data['subjects'])
                merged_data['trial_ids'] = np.array(merged_data['trial_ids'])
                
                num_total_samples = len(merged_data['labels'])
                seq_len = merged_data['eeg_mrcp_data'].shape[1] if len(merged_data['eeg_mrcp_data'].shape) > 1 else 0
                class_counts_arr = np.bincount(merged_data['labels'], minlength=2)
                
                print(f"拼接窗口 {merged_window}: {num_total_samples} 个样本。")
                print(f"  数据形状: {merged_data['eeg_mrcp_data'].shape} (应为 (n_samples, 101, 30))")
                class_names = ['Move', 'NoMove']
                for i, count in enumerate(class_counts_arr):
                    print(f"  {class_names[i]}: {count} 个样本")
            else:
                print(f"拼接窗口 {merged_window} 没有成功创建数据。")
                merged_data['eeg_mrcp_data'] = np.array([])
                merged_data['eeg_erd_data'] = np.array([])
                merged_data['labels'] = np.array([])
                merged_data['subjects'] = np.array([])
                merged_data['trial_ids'] = np.array([])

    def get_data_for_window(self, time_window_key_arg):
        """获取特定时间窗口的数据"""
        if time_window_key_arg not in self.processed_data:
            print(f"错误: 时间窗口 '{time_window_key_arg}' 在processed_data中未找到。")
            return None, None, None, None
        window_proc_data = self.processed_data[time_window_key_arg]
        if len(window_proc_data['labels']) == 0:
            print(f"错误: 时间窗口 '{time_window_key_arg}' 没有准备好的标签。")
            return None, None, None, None
            
        y_data_labels = window_proc_data['labels']
        trial_ids = window_proc_data['trial_ids']
        X_eeg_mrcp_out = window_proc_data['eeg_mrcp_data']
        X_eeg_erd_out = window_proc_data['eeg_erd_data']
        
        return X_eeg_mrcp_out, X_eeg_erd_out, y_data_labels, trial_ids

#---------------------------------------
# 增强版数据增强类
#---------------------------------------
class EEGDataAugmentor:
    """专门针对EEG数据的增强类，提供多种数据增强策略"""
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def gaussian_noise(self, eeg_data, std_range=(0.01, 0.03)):
        """添加高斯噪声"""
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, eeg_data.shape)
        return eeg_data + noise
    
    def time_shift(self, eeg_data, max_shift=5):
        """时间轴随机移位"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return eeg_data
        
        # 沿时间轴移位
        shifted_data = np.zeros_like(eeg_data)
        if shift > 0:
            shifted_data[:, shift:, :] = eeg_data[:, :-shift, :]
            # 填充开始部分，使用镜像反射
            shifted_data[:, :shift, :] = eeg_data[:, shift-1::-1, :]
        else:
            shift = abs(shift)
            shifted_data[:, :-shift, :] = eeg_data[:, shift:, :]
            # 填充结束部分，使用镜像反射
            shifted_data[:, -shift:, :] = eeg_data[:, -1:-shift-1:-1, :]
        
        return shifted_data
    
    def channel_dropout(self, eeg_data, drop_prob=0.1, channel_axis=2):
        """随机丢弃部分通道"""
        mask = np.random.binomial(1, 1-drop_prob, eeg_data.shape[channel_axis])
        mask = np.ones(eeg_data.shape)
        
        for i in range(eeg_data.shape[0]):
            for c in range(eeg_data.shape[channel_axis]):
                if np.random.rand() < drop_prob:
                    # 将整个通道置为平均值
                    if channel_axis == 2:  # 通道在最后一个维度
                        mask[i, :, c] = 0
                    else:  # 通道在第二个维度
                        mask[i, c, :] = 0
        
        return eeg_data * mask
    
    def frequency_perturbation(self, eeg_data, max_scale=0.2):
        """频率轴扰动，通过FFT后调整频谱实现"""
        perturbed_data = np.zeros_like(eeg_data)
        
        for i in range(eeg_data.shape[0]):
            for c in range(eeg_data.shape[2]):
                # 获取单个通道信号
                signal = eeg_data[i, :, c]
                
                # 应用FFT
                fft_data = np.fft.rfft(signal)
                
                # 随机缩放频率
                scale = 1 + np.random.uniform(-max_scale, max_scale)
                
                # 缩放频谱
                freq_len = len(fft_data)
                scaled_fft = np.zeros(freq_len, dtype=complex)
                
                if scale > 1:  # 扩展
                    src_idx = np.round(np.arange(0, freq_len) / scale).astype(int)
                    src_idx = src_idx[src_idx < freq_len]
                    scaled_fft[:len(src_idx)] = fft_data[src_idx]
                else:  # 压缩
                    dst_idx = np.round(np.arange(0, freq_len) * scale).astype(int)
                    dst_idx = dst_idx[dst_idx < freq_len]
                    scaled_fft[dst_idx] = fft_data[:len(dst_idx)]
                
                # 逆FFT回时域
                perturbed_signal = np.fft.irfft(scaled_fft, len(signal))
                perturbed_data[i, :, c] = perturbed_signal
        
        return perturbed_data
    
    def bandpower_augmentation(self, eeg_data, fs=250):
        """增强特定频段的功率，如μ和β节律"""
        from scipy import signal as sp_signal  # 直接导入scipy.signal
        
        augmented_data = np.copy(eeg_data)
        
        # 定义感兴趣的频段 (μ,β节律与运动相关)
        bands = {
            'mrcp':(0.1 , 5),
            'mu': (8, 12),    # μ节律 (8-12 Hz)
            'beta': (13, 30)  # β节律 (13-30 Hz)
        }
        
        for i in range(eeg_data.shape[0]):
            # 随机选择要增强的频段
            band_name = np.random.choice(list(bands.keys()))
            low, high = bands[band_name]
            
            # 随机增强系数
            scale = np.random.uniform(1.1, 1.5)
            
            for c in range(eeg_data.shape[2]):
                # 获取单个通道信号
                signal_data = eeg_data[i, :, c]
                
                # 设计带通滤波器
                nyq = fs / 2
                low_norm = low / nyq
                high_norm = high / nyq
                
                if low_norm < 1 and high_norm < 1:
                    try:
                        # 使用scipy.signal模块而不是signal属性
                        b, a = sp_signal.butter(4, [low_norm, high_norm], btype='band')
                        band_signal = sp_signal.filtfilt(b, a, signal_data)
                        augmented_data[i, :, c] = signal_data + (scale - 1) * band_signal
                    except Exception as e:
                        print(f"带通滤波器错误: {e}")
                        # 出错时保持原始信号不变
                        pass
        
        return augmented_data
    
    def spatial_interpolation(self, eeg_data, channel_axis=2, max_interp=3):
        """空间插值，随机选择通道进行插值"""
        interpolated_data = np.copy(eeg_data)
        n_channels = eeg_data.shape[channel_axis]
        
        for i in range(eeg_data.shape[0]):
            # 随机选择要插值的通道数量
            n_interp = np.random.randint(1, min(max_interp + 1, n_channels))
            
            # 随机选择要插值的通道
            interp_channels = np.random.choice(n_channels, n_interp, replace=False)
            
            for c in interp_channels:
                # 选择两个随机通道进行线性插值
                src_channels = np.random.choice([ch for ch in range(n_channels) if ch != c], 2, replace=False)
                weights = np.random.dirichlet(np.ones(2))
                
                # 进行插值
                if channel_axis == 2:  # 通道在最后一个维度
                    interpolated_data[i, :, c] = (weights[0] * eeg_data[i, :, src_channels[0]] +
                                               weights[1] * eeg_data[i, :, src_channels[1]])
                else:  # 通道在第二个维度
                    interpolated_data[i, c, :] = (weights[0] * eeg_data[i, src_channels[0], :] +
                                               weights[1] * eeg_data[i, src_channels[1], :])
        
        return interpolated_data
    
    def slope_augmentation(self, eeg_data, max_slope=0.01):
        """添加线性趋势"""
        augmented_data = np.copy(eeg_data)
        seq_len = eeg_data.shape[1]
        
        for i in range(eeg_data.shape[0]):
            # 随机斜率
            slope = np.random.uniform(-max_slope, max_slope)
            
            # 生成线性趋势
            trend = np.arange(seq_len) * slope
            
            # 应用到所有通道
            for c in range(eeg_data.shape[2]):
                augmented_data[i, :, c] = eeg_data[i, :, c] + trend
        
        return augmented_data
    
    def apply_random_augmentations(self, eeg_data, num_augmentations=2):
        """应用随机选择的增强方法"""
        augmentation_methods = [
            self.gaussian_noise,
            self.time_shift,
            self.channel_dropout,
            self.frequency_perturbation,
            self.bandpower_augmentation,
            self.spatial_interpolation,
            self.slope_augmentation
        ]
        
        # 随机选择几种增强方法
        selected_methods = np.random.choice(
            augmentation_methods, 
            size=min(num_augmentations, len(augmentation_methods)),
            replace=False
        )
        
        # 依次应用所选方法
        augmented_data = eeg_data.copy()
        for method in selected_methods:
            augmented_data = method(augmented_data)
        
        return augmented_data
    
    def augment_batch(self, eeg_data, labels, augmentation_factor=1):
        """批量增强数据"""
        n_samples = eeg_data.shape[0]
        n_new_samples = int(n_samples * augmentation_factor)
        
        if n_new_samples <= 0:
            return eeg_data, labels
        
        # 创建增强数据
        augmented_eeg = np.zeros((n_new_samples,) + eeg_data.shape[1:], dtype=eeg_data.dtype)
        augmented_labels = np.zeros(n_new_samples, dtype=labels.dtype)
        
        for i in range(n_new_samples):
            # 随机选择一个样本
            idx = np.random.randint(0, n_samples)
            sample = eeg_data[idx]
            
            # 应用随机增强
            augmented_sample = self.apply_random_augmentations(sample[np.newaxis, ...])
            
            # 保存增强样本和标签
            augmented_eeg[i] = augmented_sample[0]
            augmented_labels[i] = labels[idx]
        
        # 合并原始和增强数据
        combined_eeg = np.vstack([eeg_data, augmented_eeg])
        combined_labels = np.concatenate([labels, augmented_labels])
        
        return combined_eeg, combined_labels




========================数据增强应用（随机应用两种传统数据增强，然后采用GAN数据增强方法）=============================
def balance_and_augment_data(self, eeg_data, labels, time_window_name, augmentation_factor=2.0, use_gan=True, use_traditional=True):
        """结合多种数据增强方法平衡类别并扩充数据"""
        if not self.data_augmentation:
            return eeg_data, labels
        
        # 统计每个类别的样本数
        class_counts = np.bincount(labels, minlength=2)
        print(f"原始类别分布: {class_counts}")
        
        # 计算目标样本数
        target_per_class = int(np.max(class_counts) * augmentation_factor)
        print(f"增强后目标每类样本数: {target_per_class}")
        
        # 合并使用传统增强和GAN生成的样本
        augmented_data = eeg_data.copy()
        augmented_labels = labels.copy()
        
        # 1. 使用传统增强方法
        if use_traditional:
            print("应用传统数据增强...")
            
            for class_idx in range(len(class_counts)):
                # 当前类别的索引
                class_indices = np.where(labels == class_idx)[0]
                
                # 需要增强的样本数
                if len(class_indices) < target_per_class:
                    n_augment = min(target_per_class - len(class_indices), len(class_indices) * 3)
                    
                    if n_augment > 0:
                        print(f"为类别 {class_idx} 使用传统方法增强 {n_augment} 个样本")
                        
                        # 选择当前类别的数据
                        class_data = eeg_data[class_indices]
                        
                        # 应用传统增强
                        augmented_class_data, _ = self.data_augmentor.augment_batch(
                            class_data, 
                            np.ones(len(class_indices)) * class_idx, 
                            augmentation_factor=n_augment/len(class_indices)
                        )
                        
                        # 只取新增的部分
                        new_samples = augmented_class_data[len(class_indices):]
                        new_labels = np.ones(len(new_samples), dtype=np.int32) * class_idx
                        
                        # 合并数据
                        augmented_data = np.vstack([augmented_data, new_samples])
                        augmented_labels = np.concatenate([augmented_labels, new_labels])
        
        # 2. 使用GAN生成样本
        if use_gan:
            # 计算需要生成的样本数（每个类别）
            current_class_counts = np.bincount(augmented_labels, minlength=2)
            synthesis_counts = [max(0, target_per_class - count) for count in current_class_counts]
            
            if sum(synthesis_counts) > 0:
                print(f"需要GAN合成的样本数: {synthesis_counts}")
                
                # 确保生成器已训练
                generator_key = time_window_name.replace("_mrcp", "").replace("_erd", "")
                if generator_key not in self.eeg_generators:
                    print(f"训练 {generator_key} 的GAN...")
                    self.train_gan(generator_key, eeg_data, labels, n_epochs=70)
                
                # 生成合成数据
                synthetic_data, synthetic_labels = self.generate_synthetic_data(
                    generator_key, synthesis_counts
                )
                
                if synthetic_data is not None:
                    # 合并数据
                    augmented_data = np.vstack([augmented_data, synthetic_data])
                    augmented_labels = np.concatenate([augmented_labels, synthetic_labels])
                else:
                    print("GAN合成数据失败")
        
        print(f"增强后数据大小: {len(augmented_labels)} 个样本")
        print(f"增强后类别分布: {np.bincount(augmented_labels, minlength=2)}")
        
        return augmented_data, augmented_labels
    
    def mixup_augmentation(self, x, y, alpha=0.1):
        """MixUp数据增强 - 修复版，使用更小的alpha避免过度混合"""
        batch_size = x.size(0)
        
        # 确保不会有尺寸不匹配问题
        indices = torch.randperm(batch_size).to(x.device)
        
        # 生成混合权重
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        lam = max(lam, 1-lam)  # 确保主样本占比更高
        
        # 混合数据和标签
        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]
        
        # 返回的数据和标签应保持相同的批次大小
        assert mixed_x.size(0) == y_a.size(0) == y_b.size(0), "批次大小不匹配！"
        
        return mixed_x, y_a, y_b, lam



=========================数据标准化==========================
def normalize_data(self, data, fit=False):
        """改进的标准化方法，对每个通道独立应用"""
        shape = data.shape
        
        # 对每个通道独立标准化
        normalized_data = np.zeros_like(data)
        
        for i in range(shape[2]):  # 遍历通道
            channel_data = data[:, :, i]
            flattened = channel_data.reshape(-1, 1)
            
            if fit:
                if not hasattr(self, 'channel_scalers'):
                    self.channel_scalers = {}
                scaler = StandardScaler()
                normalized = scaler.fit_transform(flattened)
                self.channel_scalers[i] = scaler
            else:
                if hasattr(self, 'channel_scalers') and i in self.channel_scalers:
                    normalized = self.channel_scalers[i].transform(flattened)
                else:
                    normalized = flattened
            
            normalized_data[:, :, i] = normalized.reshape(channel_data.shape)
        
        return normalized_data



=============================GAN数据生成方法==================================

def train_gan(self, time_window_name, eeg_data, labels, n_epochs=70, batch_size=16, latent_dim=100):
        """训练改进的GAN用于EEG数据生成"""
        print(f"为时间窗口 {time_window_name} 训练GAN模型...")
        print(f"输入数据形状: {eeg_data.shape}, 标签数量: {len(labels)}")
        print(f"标签分布: {np.bincount(labels, minlength=2)}")
        
        # 标准化EEG数据
        eeg_data_normalized = self.normalize_data(eeg_data, fit=True)
        
        # 转换为Torch tensors
        eeg_tensor = torch.FloatTensor(eeg_data_normalized).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        # 创建数据集和加载器
        dataset = TensorDataset(eeg_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # 初始化改进的GAN模型
        seq_len, channels = eeg_data.shape[1], eeg_data.shape[2]
        generator = EnhancedEEGGenerator(latent_dim, seq_len, channels, 2).to(self.device)
        discriminator = EnhancedEEGDiscriminator(seq_len, channels, 2).to(self.device)
        
        # 优化器
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # 损失函数
        adversarial_loss = nn.BCELoss()
        
        # 训练循环
        best_g_loss = float('inf')
        
        for epoch in range(n_epochs):
            gen_losses = []
            disc_losses = []
            
            for i, (real_samples, batch_labels) in enumerate(dataloader):
                batch_size_actual = real_samples.size(0)
                
                # 真实和虚假标签
                valid = torch.ones(batch_size_actual, 1).to(self.device)
                fake = torch.zeros(batch_size_actual, 1).to(self.device)
                
                # -----------------
                # 训练判别器
                # -----------------
                optimizer_D.zero_grad()
                
                # 真实样本损失
                real_validity = discriminator(real_samples, batch_labels)
                real_loss = adversarial_loss(real_validity, valid)
                
                # 生成虚假样本
                z = torch.randn(batch_size_actual, latent_dim).to(self.device)
                fake_samples = generator(z, batch_labels)
                
                # 虚假样本损失
                fake_validity = discriminator(fake_samples.detach(), batch_labels)
                fake_loss = adversarial_loss(fake_validity, fake)
                
                # 总判别器损失
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                
                # -----------------
                # 训练生成器
                # -----------------
                if i % 2 == 0:  # 降低生成器训练频率
                    optimizer_G.zero_grad()
                    
                    # 生成新的虚假样本
                    z = torch.randn(batch_size_actual, latent_dim).to(self.device)
                    fake_samples = generator(z, batch_labels)
                    
                    # 生成器损失
                    fake_validity = discriminator(fake_samples, batch_labels)
                    g_loss = adversarial_loss(fake_validity, valid)
                    
                    # 添加特征匹配损失，使生成的数据具有更真实的特征分布
                    # 提取真实样本的特征
                    real_features = discriminator.pattern_extractor(real_samples)
                    fake_features = discriminator.pattern_extractor(fake_samples)
                    
                    # 计算特征匹配损失
                    feature_matching_loss = F.mse_loss(
                        fake_features.mean(dim=0), 
                        real_features.mean(dim=0).detach()
                    )
                    
                    # 合并损失
                    total_g_loss = g_loss + 10 * feature_matching_loss
                    
                    total_g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    optimizer_G.step()
                    
                    gen_losses.append(total_g_loss.item())
                
                disc_losses.append(d_loss.item())
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                avg_g_loss = np.mean(gen_losses) if gen_losses else 0
                avg_d_loss = np.mean(disc_losses)
                print(f"[时间窗口 {time_window_name}] Epoch {epoch+1}/{n_epochs} - "
                      f"G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                
                # 保存最佳生成器
                if avg_g_loss < best_g_loss and avg_g_loss > 0:
                    best_g_loss = avg_g_loss
                    best_generator_state = generator.state_dict()
        
        # 加载最佳状态
        if 'best_generator_state' in locals():
            generator.load_state_dict(best_generator_state)
        
        # 保存生成器
        self.eeg_generators[time_window_name] = generator
        
        # 保存模型文件
        model_path = os.path.join(self.output_dir, 'models', f'generator_{time_window_name}.pth')
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
        }, model_path)
        
        print(f"GAN模型训练完成，生成器已保存到 {model_path}")
        
        return generator
    
    def generate_synthetic_data(self, time_window_name, class_sizes, latent_dim=100, quality_check=True):
        """生成高质量的合成EEG数据"""
        if time_window_name not in self.eeg_generators:
            print(f"错误: 未找到 {time_window_name} 的生成器")
            return None, None
        
        generator = self.eeg_generators[time_window_name].eval()
        generated_data = []
        generated_labels = []
        
        with torch.no_grad():
            for class_idx in range(len(class_sizes)):
                n_samples = class_sizes[class_idx]
                if n_samples <= 0:
                    continue
                
                class_name = self.class_names[class_idx]
                print(f"为 {class_name} 生成 {n_samples} 个合成样本...")
                
                # 批量生成
                batch_size = min(32, n_samples)
                generated_count = 0
                
                while generated_count < n_samples:
                    current_batch = min(batch_size, n_samples - generated_count)
                    
                    # 生成标签
                    labels = torch.LongTensor([class_idx] * current_batch).to(self.device)
                    
                    # 生成噪声
                    z = torch.randn(current_batch, latent_dim).to(self.device)
                    
                    # 生成样本
                    fake_samples = generator(z, labels)
                    fake_samples = fake_samples.cpu().numpy()
                    
                    # 质量检查
                    if quality_check:
                        valid_mask = ~np.isnan(fake_samples).any(axis=(1, 2))
                        max_values = np.max(np.abs(fake_samples), axis=(1, 2))
                        valid_mask &= (max_values < 10.0)  # 过滤极端值
                        
                        fake_samples = fake_samples[valid_mask]
                        
                        if len(fake_samples) > 0:
                            # 应用平滑处理和时间相关性增强
                            for i in range(len(fake_samples)):
                                for c in range(fake_samples.shape[2]):
                                    # 使用高斯滤波平滑信号
                                    fake_samples[i, :, c] = gaussian_filter1d(
                                        fake_samples[i, :, c], sigma=1.0
                                    )
                                    
                                    # 确保生理合理性
                                    # 限制信号在合理范围内
                                    signal_range = np.max(np.abs(fake_samples[i, :, c]))
                                    if signal_range > 5.0:
                                        scale_factor = 5.0 / signal_range
                                        fake_samples[i, :, c] *= scale_factor
                            
                            generated_data.append(fake_samples)
                            generated_labels.extend([class_idx] * len(fake_samples))
                            generated_count += len(fake_samples)
                    else:
                        generated_data.append(fake_samples)
                        generated_labels.extend([class_idx] * current_batch)
                        generated_count += current_batch
        
        if not generated_data:
            print("没有生成合成数据")
            return None, None
        
        # 合并数据
        generated_data = np.vstack(generated_data)
        generated_labels = np.array(generated_labels)
        
        print(f"成功生成 {len(generated_labels)} 个合成样本")
        print(f"类别分布: {np.bincount(generated_labels, minlength=2)}")
        
        return generated_data, generated_labels

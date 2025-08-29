from torch import Tensor
import torch

def softmax(x: Tensor,i: int) -> Tensor:
    # 获取第i个纬度的最大值张量
    max_vals = x.max(dim=i, keepdim=True).values
    # 计算指数
    exp_x = torch.exp(x - max_vals)
    # 计算softmax
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def cross_entropy(logits: Tensor,targets: Tensor) -> Tensor:
    """
    入参：
        logits（o_i）: (...,vocab_size)
        targets(x_{i+1}): (...)
    出参：
        loss: (1,)
    """
    # 1. 确定vocab_size维度的位置
    vocab_dim = -1

    # 原生实现
    # 2. 减去logits最大值以保证数值稳定性（避免exp运算溢出）
    logits_stable = logits - logits.max(dim=vocab_dim, keepdim=True).values

    # 3. 公式： loss = avg(-log(p_θ(targets | x_{1:i})))
    # -> avg(-log(softmax(logits_stable)[targets]))
    # -> avg(-log(exp(logits_stable)[targets] / sum(exp(logits_stable)))) # 此处有两个公式 log(a/b) = log(a) - log(b), log(exp(x)) = x
    # -> avg(-logits_stable[targets] + log(sum(exp(logits_stable))))
    
    # 3.1 计算 log(sum(exp(logits_stable)))
    exp_logits = torch.exp(logits_stable)
    sum_exp_logits = torch.sum(exp_logits, dim=vocab_dim, keepdim=True)
    log_sum_exp_logits = torch.log(sum_exp_logits)

    # 3.2 计算 logits_stable[targets]
    # 扩维并提取目标值
    targets_reshaped = targets.unsqueeze(dim=vocab_dim)
    logits_target  = logits_stable.gather(dim=vocab_dim, index=targets_reshaped)

    # 3.3 计算单个样本的损失 -logits_stable[targets] + log(sum(exp(logits_stable)))
    per_sample_loss = -logits_target + log_sum_exp_logits

    # 3.4 计算平均损失 avg(-logits_stable[targets] + log(sum(exp(logits_stable))))
    average_loss = per_sample_loss.mean()

    return average_loss
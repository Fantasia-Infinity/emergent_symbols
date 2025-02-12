import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 防止Python生成.pyc文件
sys.dont_write_bytecode = True
# 将models和tasks目录添加到系统路径
sys.path.insert(0, './models')
sys.path.insert(0, './tasks')
# 导入日志工具
from util import log

# 检查路径是否存在，不存在则创建
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class seq_dataset(Dataset):
    """序列数据集类，继承自PyTorch的Dataset类"""
    def __init__(self, dset, args):
        """初始化数据集
        Args:
            dset (dict): 包含序列索引和标签的数据集
            args: 命令行参数
        """
        self.seq_ind = dset['seq_ind']  # 序列索引
        self.y = dset['y']  # 标签
        self.len = self.seq_ind.shape[0]  # 数据集长度
    
    def __len__(self):
        """返回数据集长度"""
        return self.len
    
    def __getitem__(self, idx):
        """获取单个样本
        Args:
            idx (int): 样本索引
        Returns:
            tuple: (序列索引, 标签)
        """
        seq_ind = self.seq_ind[idx]
        y = self.y[idx]
        return seq_ind, y

def train(args, model, device, optimizer, epoch, all_imgs, train_loader):
    """训练模型
    Args:
        args: 命令行参数
        model: 要训练的模型
        device: 使用的设备 (CPU/GPU)
        optimizer: 优化器
        epoch: 当前epoch
        all_imgs: 所有图像数据
        train_loader: 训练数据加载器
    """
    # 创建保存训练进度的目录
    train_prog_dir = './train_prog/'
    check_path(train_prog_dir)
    task_dir = train_prog_dir + args.task + '/'
    check_path(task_dir)
    gen_dir = task_dir + 'm' + str(args.m_holdout) + '/'
    check_path(gen_dir)
    model_dir = gen_dir + args.model_name + '/'
    check_path(model_dir)
    run_dir = model_dir + 'run' + args.run + '/'
    check_path(run_dir)
    train_prog_fname = run_dir + 'epoch_' + str(epoch) + '.txt'
    train_prog_f = open(train_prog_fname, 'w')
    train_prog_f.write('batch loss acc\n')
    
    # 设置为训练模式
    model.train()
    
    # 遍历批次
    for batch_idx, (seq_ind, y) in enumerate(train_loader):
        # 记录批次开始时间
        start_time = time.time()
        # 使用序列索引获取对应图像
        x_seq = all_imgs[seq_ind,:,:]
        # 将数据加载到设备
        x_seq = x_seq.to(device)
        y = y.to(device)
        # 清空优化器梯度
        optimizer.zero_grad()
        # 运行模型
        y_pred_linear, y_pred, _, __ = model(x_seq, device)
        # 计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred_linear, y)
        # 更新模型
        loss.backward()
        optimizer.step()
        # 计算批次耗时
        end_time = time.time()
        batch_dur = end_time - start_time
        # 报告进度
        if batch_idx % args.log_interval == 0:
            # 计算准确率
            acc = torch.eq(y_pred, y).float().mean().item() * 100.0
            # 记录日志
            log.info('[Epoch: ' + str(epoch) + '] ' + \
                     '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
                     '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
                     '[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
                     '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
            # 保存进度到文件
            train_prog_f.write(str(batch_idx) + ' ' +\
                               '{:.4f}'.format(loss.item()) + ' ' + \
                               '{:.2f}'.format(acc) + '\n')
    train_prog_f.close()

def extract_reps(args, model, device, all_imgs, test_loader, train_test='train'):
    """提取学习到的表示
    Args:
        args: 命令行参数
        model: 训练好的模型
        device: 使用的设备 (CPU/GPU)
        all_imgs: 所有图像数据
        test_loader: 测试数据加载器
        train_test (str): 数据集类型 ('train' 或 'test')
    """
    log.info('Extracting learned representations for ' + train_test + ' set...')
    # 设置为评估模式
    model.eval()
    # 初始化存储变量
    all_acc = []
    all_loss = []
    all_M_k = []
    all_key_r = []
    all_seq = []
    # 遍历批次
    for batch_idx, (seq_ind, y) in enumerate(test_loader):
        # 使用序列索引获取对应图像
        x_seq = all_imgs[seq_ind,:,:]
        # 将数据加载到设备
        x_seq = x_seq.to(device)
        y = y.to(device)
        # 运行模型
        y_pred_linear, y_pred, M_k, key_r = model(x_seq, device)
        all_M_k.append(M_k)
        all_key_r.append(key_r)
        all_seq.append(seq_ind.cpu().numpy())
        # 计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred_linear, y)
        all_loss.append(loss.item())
        # 计算准确率
        acc = torch.eq(y_pred, y).float().mean().item() * 100.0
        all_acc.append(acc)
        # 报告进度
        log.info('[Batch: ' + str(batch_idx) + ' of ' + str(len(test_loader)) + ']')
    # 报告整体测试性能
    avg_loss = np.mean(all_loss)
    avg_acc = np.mean(all_acc)
    log.info('[Summary] ' + \
             '[Loss = ' + '{:.4f}'.format(avg_loss) + '] ' + \
             '[Accuracy = ' + '{:.2f}'.format(avg_acc) + ']')
    # 合并所有批次的表示
    all_M_k = np.concatenate(all_M_k, 0)
    all_key_r = np.concatenate(all_key_r, 0)
    all_seq = np.concatenate(all_seq, 0)
    # 保存到文件
    reps_dir = './extracted_reps/'
    check_path(reps_dir)
    reps_fname = reps_dir + 'run' + args.run + '_' + train_test + '_reps.npz'
    np.savez(reps_fname, M_k=all_M_k, all_key_r=all_key_r, seq=all_seq)

def main():
    """主函数，处理命令行参数并执行训练和特征提取"""
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    # 模型设置
    parser.add_argument('--model_name', type=str, default='ESBN_return_keys', help="仅兼容'ESBN_return_keys'模型")
    parser.add_argument('--norm_type', type=str, default='contextnorm', help="{'nonorm', 'contextnorm', 'tasksegmented_contextnorm'}")
    parser.add_argument('--encoder', type=str, default='conv', help="{'conv', 'mlp', 'rand'}")
    # 任务设置
    parser.add_argument('--task', type=str, default='dist3', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")
    parser.add_argument('--train_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
    parser.add_argument('--test_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
    parser.add_argument('--n_shapes', type=int, default=100, help="n = 训练和测试可用的总形状数")
    parser.add_argument('--m_holdout', type=int, default=95, help="m = 训练期间保留的对象数（从n中）")
    # 训练设置
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--train_set_size', type=int, default=10000)
    parser.add_argument('--train_proportion', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_interval', type=int, default=10)
    # 测试设置
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--test_set_size', type=int, default=10000)
    # 设备设置
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    # 运行编号
    parser.add_argument('--run', type=str, default='1')
    args = parser.parse_args()

    # 设置CUDA
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 随机分配对象到训练集或测试集
    all_shapes = np.arange(args.n_shapes)
    np.random.shuffle(all_shapes)
    if args.m_holdout > 0:
        train_shapes = all_shapes[args.m_holdout:]
        test_shapes = all_shapes[:args.m_holdout]
    else:
        train_shapes = all_shapes
        test_shapes = all_shapes
    
    # 生成训练和测试集
    task_gen = __import__(args.task)
    log.info('Generating task: ' + args.task + '...')
    args, train_set, test_set = task_gen.create_task(args, train_shapes, test_shapes)
    
    # 转换为PyTorch DataLoader
    train_set = seq_dataset(train_set, args)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    test_set = seq_dataset(test_set, args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    # 加载图像
    all_imgs = []
    for i in range(args.n_shapes):
        img_fname = './imgs/' + str(i) + '.png'
        img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs, 0)

    # 创建模型
    model_class = __import__(args.model_name)
    model = model_class.Model(task_gen, args).to(device)

    # 将相关超参数值附加到模型名称
    args.model_name = args.model_name + '_' + args.norm_type + '_lr' + str(args.lr)

    # 创建优化器
    log.info('Setting up optimizer...')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    log.info('Training begins...')
    for epoch in range(1, args.epochs + 1):
        # 训练循环
        train(args, model, device, optimizer, epoch, all_imgs, train_loader)

    # 提取训练集表示
    extract_reps(args, model, device, all_imgs, train_loader, train_test='train')
    # 提取测试集表示
    extract_reps(args, model, device, all_imgs, test_loader, train_test='test')

if __name__ == '__main__':
    main()

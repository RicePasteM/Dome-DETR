import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    ap_dict = {}
    current_epoch = None

    # 用于匹配关键信息的正则表达式
    epoch_pattern = re.compile(r'Epoch: \[(\d+)\] Total time:')
    ap_pattern = re.compile(r' Average Precision  \(AP\) \@\[ IoU\=0.50:0.95 \| area\=   all \| maxDets\=1500 \] \= (.*)')

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配epoch结束行（包含Total time）
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # 匹配AP数值行
            ap_match = ap_pattern.search(line)
            if ap_match and current_epoch is not None:
                ap = float(ap_match.group(1))
                ap_dict[current_epoch] = ap
                current_epoch = None  # 重置当前epoch

    return ap_dict


def plot_ap(ap_dict, save_path, max_epoch=159):
    # 生成0-159的epoch列表
    epochs = list(range(max_epoch + 1))
    aps = [ap_dict.get(e, None) for e in epochs]
    # 找到最高点的epoch和AP值
    max_ap = max([ap for ap in aps if ap is not None])  # 过滤掉None值
    max_ap_epoch = epochs[max(loc for loc, val in enumerate(aps) if val == max_ap)]

    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, aps, 'b-o', markersize=4)

    # 绘制最高点的虚线
    plt.axvline(x=max_ap_epoch, color='r', linestyle='--', label=f'Max AP at Epoch {max_ap_epoch}')
    plt.axhline(y=max_ap, color='g', linestyle='--', label=f'Max AP {max_ap:.3f}')

    plt.title('Average Precision (AP) @[IoU=0.50:0.95]')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.grid(True)
    plt.xlim(0, max_epoch)
    plt.ylim(0, 0.4)  # 根据实际情况调整y轴范围
    plt.legend()
    plt.savefig(save_path)


# 使用示例
if __name__ == "__main__":
    log_file = "logs/20250401.log"  # 替换为你的日志文件路径
    ap_data = parse_log(log_file)
    plot_ap(ap_data, log_file.split(".")[0], max_epoch=160)
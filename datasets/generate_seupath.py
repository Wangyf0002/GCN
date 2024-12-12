import argparse
import os
import sys

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 现在可以正常导入 SEUPath 和 PathGraph
from datasets.PathGraph import PathGraph
from datasets.SEUPath import SEUPath

def main(args):
    # 初始化 SEUPath 数据处理类
    seu_path = SEUPath(
        sample_length=args.sample_length,
        data_dir=args.data_dir,
        InputType=args.input_type,
        task=args.task
    )

    # 执行数据准备过程
    if args.test:
        test_data = seu_path.data_preprare(test=True)
        print(f"测试集已生成，共 {len(test_data)} 个样本。")
    else:
        train_data, val_data = seu_path.data_preprare()
        print(f"训练集和验证集已生成：")
        print(f"  - 训练集样本数: {len(train_data)}")
        print(f"  - 验证集样本数: {len(val_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SEUPath Dataset")
    parser.add_argument("--sample_length", type=int, default=1024, help="每个样本的长度 (默认: 1024)")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录")
    parser.add_argument("--input_type", type=str, choices=["TD", "FD"], required=True, help="输入数据类型 ('TD' 或 'FD')")
    parser.add_argument("--task", type=str, choices=["Node", "Graph"], required=True, help="任务类型 ('Node' 或 'Graph')")
    parser.add_argument("--test", action="store_true", help="仅生成测试集")
    args = parser.parse_args()

    main(args)

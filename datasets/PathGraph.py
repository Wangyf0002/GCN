from datasets.Generator import Gen_graph  # 导入生成图的工具函数

def PathGraph(interval, data, label, task):
    """
    将一维信号数据分段并生成图数据集
    参数:
    - interval: 每张图的节点数
    - data: 输入信号数据列表
    - label: 当前数据的类别标签
    - task: 数据任务信息
    返回:
    - graphset: 图数据集，生成的图以节点形式组织信号段
    """
    a, b = 0, interval  # 初始化滑窗起点和终点
    graph_list = []  # 存储生成的图数据节点列表

    # 使用滑窗方式按 interval 分段信号
    while b <= len(data):
        # 将从 a 到 b 的信号片段分为一个图的节点集
        graph_list.append(data[a:b])
        a += interval  # 滑窗起点右移
        b += interval  # 滑窗终点右移

    # 调用 Gen_graph 工具生成图数据集
    graphset = Gen_graph("PathGraph", graph_list, label, task)
    return graphset  # 返回生成的图数据集

import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告（如Plotly/Seaborn的版本警告）

import pandas as pd
import seaborn as sns
import matplotlib
# 可选：强制使用Agg后端（适配无GUI环境，注释掉则启用交互式后端）
# matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict

# ====================== 全局参数配置区（一键调整） ======================
# 基础配置
DATASET_NAME = 'iris'  # 数据集名称（Seaborn内置）
SAVE_BOXPLOT_PATH = "iris_boxplots.png"  # 箱线图保存路径（None则不保存）

# 可视化配置
FIG_SIZE = (15, 12)  # 箱线图画布尺寸
FIG_DPI = 300  # 箱线图分辨率
BOXPLOT_PALETTE = ['#FF4444', '#0066CC', '#90EE90']  # 三类鸢尾花配色（红/蓝/绿）
PLOTLY_COLOR_SEQUENCE = ['red', 'blue', 'green']  # Plotly散点图配色
TITLE_FONT_SIZE = 14  # 子图标题字体大小
LABEL_FONT_SIZE = 12  # 坐标轴标签字体大小

# 特征配置
NUMERIC_FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # 数值特征列表
CATEGORY_COL = 'species'  # 类别列名

def load_iris_data() -> pd.DataFrame:
    """
    加载Seaborn内置的鸢尾花数据集
    返回：原始DataFrame
    """
    try:
        df_iris = sns.load_dataset(DATASET_NAME)
        print(f"✅ 数据集加载完成：")
        print(f"  - 数据维度：{df_iris.shape[0]}行 × {df_iris.shape[1]}列")
        print(f"  - 原始类别：{df_iris[CATEGORY_COL].unique().tolist()}")
        print(f"  - 前5行数据：\n{df_iris.head()}")
        print(f"  - 索引50-99行数据：\n{df_iris.iloc[50:100]}")
        return df_iris
    except Exception as e:
        raise RuntimeError(f"❌ 数据集加载失败：{str(e)}")

def preprocess_iris_data(df_iris: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理：删除缺失值 + 类别列编码（保留原始标签映射）
    返回：预处理后的DataFrame（新增species_name列保留原始名称）
    """
    # 1. 缺失值处理
    print(f"\n📊 缺失值统计：\n{df_iris.isnull().sum()}")
    df_processed = df_iris.dropna()
    if len(df_processed) < len(df_iris):
        print(f"⚠️ 删除了{len(df_iris)-len(df_processed)}行缺失值数据")
    else:
        print("✅ 无缺失值，无需删除")
    
    # 2. 特征存在性校验
    missing_features = [feat for feat in NUMERIC_FEATURES if feat not in df_processed.columns]
    if missing_features:
        raise ValueError(f"❌ 缺失特征：{missing_features}，请检查特征名")
    
    # 3. 类别列编码（保留原始名称映射）
    df_processed['species_name'] = df_processed[CATEGORY_COL]  # 保留原始名称
    df_processed[CATEGORY_COL] = df_processed[CATEGORY_COL].astype('category').cat.codes
    # 打印类别映射关系
    species_mapping = df_processed[['species', 'species_name']].drop_duplicates().sort_values('species')
    print(f"\n🔍 类别编码映射：\n{species_mapping.to_string(index=False)}")
    
    return df_processed

def plot_static_boxplots(df_processed: pd.DataFrame):
    """
    绘制2×2静态箱线图（Seaborn+Matplotlib），匹配原代码布局
    """
    # 创建2行2列子图
    fig, ax_array = plt.subplots(2, 2, figsize=FIG_SIZE, dpi=FIG_DPI)
    ax_array = ax_array.flatten()  # 展平便于循环
    
    # 循环绘制每个特征的箱线图（替代重复代码）
    for idx, feature in enumerate(NUMERIC_FEATURES):
        sns.boxplot(
            x=CATEGORY_COL,
            y=feature,
            data=df_processed,
            ax=ax_array[idx],
            palette=BOXPLOT_PALETTE,
            linewidth=1.2  # 线条宽度，提升美观度
        )
        # 设置子图标题和标签样式
        ax_array[idx].set_title(f'{feature.replace("_", " ").title()} by Species', fontsize=TITLE_FONT_SIZE)
        ax_array[idx].set_xlabel('Species (0=Setosa, 1=Versicolor, 2=Virginica)', fontsize=LABEL_FONT_SIZE)
        ax_array[idx].set_ylabel(feature.replace("_", " ").title(), fontsize=LABEL_FONT_SIZE)
        ax_array[idx].grid(alpha=0.3, axis='y')  # 仅Y轴网格，更清晰
    
    # 调整布局避免重叠
    plt.tight_layout()
    
    # 保存图片（可选）
    if SAVE_BOXPLOT_PATH:
        plt.savefig(SAVE_BOXPLOT_PATH, dpi=FIG_DPI, bbox_inches='tight')
        print(f"\n✅ 静态箱线图已保存至：{SAVE_BOXPLOT_PATH}")
    
    # 显示图表（非Agg后端时生效）
    plt.show()
    plt.close(fig)

def plot_interactive_scatterplots(df_processed: pd.DataFrame):
    """
    绘制所有特征两两组合的交互式散点图（Plotly），替代原代码6次重复调用
    """
    # 生成所有特征两两组合（无重复）
    feature_pairs = []
    for i in range(len(NUMERIC_FEATURES)):
        for j in range(i+1, len(NUMERIC_FEATURES)):
            feature_pairs.append((NUMERIC_FEATURES[i], NUMERIC_FEATURES[j]))
    
    print(f"\n📈 开始绘制{len(feature_pairs)}个交互式散点图...")
    
    # 循环绘制每个特征对的散点图
    for x_feat, y_feat in feature_pairs:
        fig = px.scatter(
            df_processed,
            x=x_feat,
            y=y_feat,
            color='species_name',  # 显示原始物种名（而非数字），更易理解
            title=f"{x_feat.replace('_', ' ').title()} vs {y_feat.replace('_', ' ').title()}",
            color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
            labels={
                'species_name': 'Species',
                x_feat: x_feat.replace('_', ' ').title(),
                y_feat: y_feat.replace('_', ' ').title()
            },
            hover_data=['species']  # 悬浮显示编码值，便于对照
        )
        # 优化图表样式
        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=12
        )
        # 显示图表
        fig.show()
    
    print("✅ 所有交互式散点图绘制完成！")

def main():
    """主函数：串联所有流程"""
    try:
        print("="*60)
        print("🚀 开始执行鸢尾花数据可视化任务")
        print("="*60)
        
        # 1. 加载原始数据
        df_iris = load_iris_data()
        
        # 2. 数据预处理
        df_processed = preprocess_iris_data(df_iris)
        
        # 3. 绘制静态箱线图
        plot_static_boxplots(df_processed)
        
        # 4. 绘制交互式散点图
        plot_interactive_scatterplots(df_processed)
        
        print("\n🎉 所有可视化任务完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()

# ============================================
# 动漫台词文本数据分析完整流程
# 版本：2.0 - 支持Excel文件
# 功能：从Excel/TXT文件到深度分析报告的全流程
# ============================================

# 第一部分：导入所有必要的库
print("正在导入库...")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP和机器学习库
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# 下载必要的NLTK数据
print("下载NLTK数据...")
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK数据下载失败，尝试使用本地缓存")

# 设置中文字体（如果需要中文显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("所有库导入完成！\n" + "="*60)


# ============================================
# 第二部分：数据加载与预处理（支持Excel和文本）
# ============================================

def load_and_clean_data(file_path, text_column='dialogue'):
    """
    加载和清洗台词数据（支持Excel和文本文件）
    
    参数:
        file_path: 文件路径（支持.txt, .xlsx, .xls, .csv）
        text_column: Excel文件中台词文本所在的列名
    """
    print(f"正在加载数据: {file_path}")
    
    import os
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 根据文件扩展名选择加载方式
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.xlsx', '.xls']:
        # Excel文件处理
        print(f"检测到Excel文件，正在读取...")
        try:
            # 尝试读取Excel文件
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"读取Excel失败: {e}")
                # 尝试不同的引擎
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except:
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except:
                        raise Exception("无法读取Excel文件，请确保已安装openpyxl或xlrd库")
            
            print(f"成功读取Excel文件，共{len(df)}行，{len(df.columns)}列")
            print(f"列名: {list(df.columns)}")
            
            # 自动检测台词列
            if text_column not in df.columns:
                print(f"警告: 找不到指定的列 '{text_column}'")
                print(f"可用列: {list(df.columns)}")
                
                # 尝试自动检测台词列
                possible_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    # 检查是否包含常见的关键词
                    if any(keyword in col_lower for keyword in 
                          ['text', 'dialogue', 'line', '台词', '对话', '脚本', '台词内容', '内容']):
                        possible_cols.append(col)
                    elif df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                        # 如果列是文本类型，检查样本内容
                        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ''
                        if isinstance(sample, str) and len(sample) > 10:
                            possible_cols.append(col)
                
                if possible_cols:
                    text_column = possible_cols[0]
                    print(f"自动选择台词列: '{text_column}'")
                else:
                    # 选择第一个文本列
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if text_cols:
                        text_column = text_cols[0]
                        print(f"使用第一个文本列: '{text_column}'")
                    else:
                        # 如果没有文本列，使用第一列
                        text_column = df.columns[0]
                        print(f"使用第一列: '{text_column}'")
            
            # 提取台词文本
            lines = df[text_column].fillna('').astype(str).tolist()
            print(f"从列 '{text_column}' 提取了 {len(lines)} 行台词")
            
            # 保存原始数据框（包含所有列）
            raw_df = df.copy()
            
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return None
            
    elif file_ext == '.txt':
        # 文本文件处理
        print("检测到文本文件，正在读取...")
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"使用 {encoding} 编码加载了 {len(lines)} 行台词")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # 所有编码都失败
                raise Exception(f"无法用任何支持的编码读取文件: {encodings}")
                
            raw_df = None
            
        except Exception as e:
            print(f"读取文本文件失败: {e}")
            return None
                
    elif file_ext == '.csv':
        # CSV文件处理
        print("检测到CSV文件，正在读取...")
        try:
            # 尝试多种编码读取CSV
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"使用 {encoding} 编码成功读取CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("无法用任何支持的编码读取CSV文件")
            
            print(f"成功读取CSV文件，共{len(df)}行，{len(df.columns)}列")
            
            # 自动检测台词列
            if text_column not in df.columns:
                possible_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in 
                          ['text', 'dialogue', 'line', '台词', '对话']):
                        possible_cols.append(col)
                
                if possible_cols:
                    text_column = possible_cols[0]
                else:
                    text_column = df.columns[0]
            
            lines = df[text_column].fillna('').astype(str).tolist()
            print(f"从列 '{text_column}' 提取了 {len(lines)} 行台词")
            raw_df = df.copy()
            
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return None
            
    else:
        print(f"错误: 不支持的文件格式: {file_ext}")
        print("支持格式: .txt, .csv, .xlsx, .xls")
        return None
    
    # 创建基础数据框
    df_base = pd.DataFrame({
        'raw_line': lines,
        'line_number': range(1, len(lines) + 1)
    })
    
    # 清洗台词
    def clean_dialogue(line):
        if not isinstance(line, str):
            return ""
        
        # 移除常见的剧本标记
        line = re.sub(r'^[A-Z\s]+:', '', line)  # 移除"CHARACTER: "
        line = re.sub(r'\(.*?\)', '', line)      # 移除括号内容
        line = re.sub(r'\[.*?\]', '', line)      # 移除方括号内容
        line = re.sub(r'<.*?>', '', line)        # 移除HTML标签
        
        # 标准化标点
        line = re.sub(r'[ ]+', ' ', line)        # 多个空格变一个
        line = re.sub(r'\.{2,}', '...', line)    # 标准化省略号
        
        # 清理引号和特殊字符
        line = line.replace('"', '').replace("'", '')
        line = re.sub(r'[^\w\s.,!?-]', '', line)
        
        return line.strip()
    
    df_base['cleaned_line'] = df_base['raw_line'].apply(clean_dialogue)
    
    # 移除空行和过短的行
    initial_count = len(df_base)
    df_base = df_base[df_base['cleaned_line'].str.len() > 3]
    df_base.reset_index(drop=True, inplace=True)
    
    print(f"清洗后有效台词数: {len(df_base)} (移除了 {initial_count - len(df_base)} 行)")
    
    # 计算基本长度指标
    df_base['char_length'] = df_base['cleaned_line'].apply(len)
    df_base['word_count'] = df_base['cleaned_line'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
    
    # 如果是从结构化文件加载的，尝试合并元数据
    if raw_df is not None and len(df_base) > 0:
        try:
            # 确保长度匹配
            if len(df_base) == len(raw_df):
                print("正在合并元数据...")
                
                # 排除已存在的列和台词列
                exclude_cols = ['raw_line', 'cleaned_line', 'char_length', 'word_count', text_column]
                
                for col in raw_df.columns:
                    if col not in exclude_cols and col not in df_base.columns:
                        # 检查列类型，确保可以合并
                        if len(df_base) == len(raw_df[col]):
                            df_base[col] = raw_df[col].values
                            print(f"  添加列: {col} ({raw_df[col].dtype})")
                        
                print(f"合并了 {len([c for c in df_base.columns if c not in ['raw_line', 'line_number', 'cleaned_line', 'char_length', 'word_count']])} 个元数据列")
        except Exception as e:
            print(f"合并元数据时出错: {e}")
    
    return df_base


# ============================================
# 第三部分：词汇统计分析
# ============================================

def analyze_vocabulary(df):
    """
    分析词汇使用情况
    """
    print("\n" + "="*60)
    print("开始词汇统计分析...")
    
    # 合并所有文本
    all_text = " ".join(df['cleaned_line'].tolist())
    
    # 基础分词
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # 统计词频
    word_freq = Counter(words)
    
    # 计算统计量
    total_words = len(words)
    unique_words = len(word_freq)
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    
    # 获取停用词
    stop_words = set(stopwords.words('english'))
    
    # 过滤停用词后的高频词
    content_words = {word: freq for word, freq in word_freq.items() 
                    if word not in stop_words}
    top_content_words = dict(sorted(content_words.items(), 
                                   key=lambda x: x[1], reverse=True)[:20])
    
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'lexical_diversity': lexical_diversity,
        'word_freq': word_freq,
        'top_content_words': top_content_words
    }
    
    print(f"总词数: {total_words:,}")
    print(f"独特词数: {unique_words:,}")
    print(f"词汇多样性: {lexical_diversity:.3%}")
    print(f"前10个高频词（排除停用词）:")
    for i, (word, freq) in enumerate(list(top_content_words.items())[:10], 1):
        print(f"  {i:2d}. {word:15s}: {freq:5d}次")
    
    return stats


# ============================================
# 第四部分：情感分析
# ============================================

def perform_sentiment_analysis(df):
    """
    执行情感分析
    """
    print("\n" + "="*60)
    print("开始情感分析...")
    
    # 初始化情感分析器
    sia = SentimentIntensityAnalyzer()
    
    # 扩展动漫相关词汇
    anime_words = {
        'sugoi': 2.5,    # すごい (厉害)
        'kawaii': 2.0,   # かわいい (可爱)
        'baka': -2.0,    # ばか (笨蛋)
        'urusai': -1.5,  # うるさい (吵死了)
        'yamete': -1.8,  # やめて (住手)
        'arigato': 1.5,  # ありがとう (谢谢)
        'gomen': -1.0,   # ごめん (对不起)
        'daijoubu': 0.5, # だいじょうぶ (没关系)
    }
    
    # 添加到词典
    for word, score in anime_words.items():
        sia.lexicon[word] = score
    
    # 分析每句台词的情感
    sentiment_results = []
    for text in df['cleaned_line']:
        scores = sia.polarity_scores(text)
        
        # 分类情感
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        sentiment_results.append({
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
    
    # 合并结果
    sentiment_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df, sentiment_df], axis=1)
    
    # 计算移动平均（平滑情感曲线）
    window_size = min(20, len(df) // 10)
    if window_size % 2 == 0:
        window_size += 1
    
    df['compound_ma'] = df['compound'].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    
    # 输出情感统计
    positive_pct = (df['sentiment'] == 'positive').sum() / len(df)
    negative_pct = (df['sentiment'] == 'negative').sum() / len(df)
    neutral_pct = (df['sentiment'] == 'neutral').sum() / len(df)
    
    print(f"积极台词: {positive_pct:.1%}")
    print(f"消极台词: {negative_pct:.1%}")
    print(f"中性台词: {neutral_pct:.1%}")
    print(f"平均情感强度: {df['compound'].abs().mean():.3f}")
    print(f"整体情感基调: ", end="")
    if df['compound'].mean() > 0.1:
        print("积极向上")
    elif df['compound'].mean() < -0.1:
        print("消极低沉")
    else:
        print("中性平衡")
    
    # 识别情感转折点
    turning_points = []
    for i in range(1, len(df) - 1):
        if pd.notna(df.loc[i, 'compound_ma']) and pd.notna(df.loc[i-1, 'compound_ma']):
            prev = df.loc[i-1, 'compound_ma']
            curr = df.loc[i, 'compound_ma']
            next_val = df.loc[i+1, 'compound_ma'] if i+1 < len(df) else curr
            
            # 检测峰值
            if prev < curr > next_val and (curr - prev) > 0.3:
                turning_points.append(('peak', i, curr, df.loc[i, 'cleaned_line'][:50]))
            # 检测谷值
            elif prev > curr < next_val and (prev - curr) > 0.3:
                turning_points.append(('valley', i, curr, df.loc[i, 'cleaned_line'][:50]))
    
    print(f"发现情感转折点: {len(turning_points)} 个")
    
    return df, turning_points


# ============================================
# 第五部分：主题建模
# ============================================

def perform_topic_modeling(df, num_topics=None):
    """
    执行主题建模
    """
    print("\n" + "="*60)
    print("开始主题建模...")
    
    # 准备文本数据
    print("预处理文本数据...")
    
    # 获取停用词并添加自定义停用词
    stop_words = list(stopwords.words('english'))
    custom_stopwords = ['oh', 'ah', 'um', 'uh', 'huh', 'hey', 'well', 
                       'just', 'like', 'really', 'maybe', 'actually',
                       'thing', 'things', 'something', 'anything']
    stop_words.extend(custom_stopwords)
    
    # 简单的文本预处理函数
    def preprocess_text(text):
        # 转为小写
        text = text.lower()
        # 移除数字和特殊字符
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词并移除停用词
        words = [word for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    df['processed_text'] = df['cleaned_line'].apply(preprocess_text)
    
    # 创建文档-词矩阵
    print("创建文档-词矩阵...")
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        max_features=1000,
        stop_words=stop_words
    )
    
    X = vectorizer.fit_transform(df['processed_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"文档-词矩阵形状: {X.shape}")
    print(f"词汇表大小: {len(feature_names)}")
    
    # 自动确定主题数量（如果没有指定）
    if num_topics is None:
        print("自动确定最佳主题数量...")
        perplexities = []
        topic_range = range(2, min(9, len(df) // 50 + 2))
        
        for n in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            lda.fit(X)
            perplexities.append(lda.perplexity(X))
        
        # 选择困惑度下降变缓的点
        if len(perplexities) > 1:
            diffs = np.diff(perplexities)
            diffs_ratio = diffs[1:] / diffs[:-1] if len(diffs) > 1 else [1]
            optimal_idx = np.argmin(diffs_ratio) + 2 if len(diffs_ratio) > 0 else 2
            num_topics = min(max(3, optimal_idx), 6)
        else:
            num_topics = 4
        
        print(f"选择主题数量: {num_topics}")
    
    # 训练LDA模型
    print(f"训练LDA模型 ({num_topics}个主题)...")
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda.fit(X)
    
    # 显示主题关键词
    def display_topics(model, feature_names, n_top_words=10):
        topics_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics_dict[f'Topic_{topic_idx+1}'] = top_words
            
            print(f"\n主题 #{topic_idx+1}:")
            print("  " + ", ".join(top_words))
        
        return topics_dict
    
    print("\nLDA主题关键词:")
    topics_dict = display_topics(lda, feature_names)
    
    # 为每句台词分配主题
    topic_distribution = lda.transform(X)
    df['dominant_topic'] = topic_distribution.argmax(axis=1)
    df['topic_confidence'] = topic_distribution.max(axis=1)
    
    # 分析主题分布
    topic_counts = df['dominant_topic'].value_counts().sort_index()
    print("\n主题分布:")
    for topic_id in range(num_topics):
        count = topic_counts.get(topic_id, 0)
        pct = count / len(df)
        print(f"  主题 {topic_id+1}: {count}句台词 ({pct:.1%})")
    
    # 分析每个主题的情感倾向
    print("\n各主题平均情感:")
    topic_sentiment = df.groupby('dominant_topic')['compound'].agg(['mean', 'std', 'count'])
    for topic_id, row in topic_sentiment.iterrows():
        sentiment = "积极" if row['mean'] > 0.1 else "消极" if row['mean'] < -0.1 else "中性"
        print(f"  主题 {topic_id+1}: {row['mean']:.3f} ({sentiment})")
    
    return df, topics_dict, lda, X, vectorizer


# ============================================
# 第六部分：可视化分析
# ============================================

def create_visualizations(df, topics_dict, turning_points, vocab_stats):
    """
    创建所有可视化图表
    """
    print("\n" + "="*60)
    print("创建可视化图表...")
    
    try:
        # 创建图形
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 情感叙事曲线
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df.index, df['compound'], alpha=0.5, linewidth=0.8, label='原始', color='lightblue')
        
        if 'compound_ma' in df.columns and df['compound_ma'].notna().sum() > 0:
            ax1.plot(df.index, df['compound_ma'], linewidth=2, label='平滑', color='red')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 标记转折点
        for point_type, idx, value, line_text in turning_points[:5]:  # 只标记前5个
            color = 'green' if point_type == 'peak' else 'red'
            marker = '^' if point_type == 'peak' else 'v'
            ax1.scatter(idx, value, color=color, marker=marker, s=100, zorder=5)
        
        ax1.set_xlabel('台词序列')
        ax1.set_ylabel('情感强度')
        ax1.set_title('情感叙事曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 情感分布直方图
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(df['compound'], bins=30, edgecolor='black', alpha=0.7, color='skyblue', orientation='horizontal')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('频率')
        ax2.set_ylabel('情感强度')
        ax2.set_title('情感强度分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 台词长度分布
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(df['word_count'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax3.axvline(df['word_count'].mean(), color='red', linestyle='--', 
                    label=f'平均: {df["word_count"].mean():.1f}')
        ax3.set_xlabel('台词词数')
        ax3.set_ylabel('频率')
        ax3.set_title('台词长度分布（按词数）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 主题随时间分布
        ax4 = plt.subplot(3, 3, 4)
        num_topics = len(topics_dict)
        colors = plt.cm.Set3(np.linspace(0, 1, num_topics))
        
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            ax4.scatter(df[topic_mask].index, 
                       [topic_id + 1] * len(df[topic_mask]),
                       alpha=0.4, s=10, color=colors[topic_id],
                       label=f'主题 {topic_id+1}')
        
        ax4.set_xlabel('台词序列')
        ax4.set_ylabel('主题')
        ax4.set_title('主题随时间分布')
        ax4.set_yticks(range(1, num_topics + 1))
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. 主题-情感关系
        ax5 = plt.subplot(3, 3, 5)
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            ax5.scatter(df[topic_mask].index, 
                       df[topic_mask]['compound'],
                       alpha=0.5, s=15, color=colors[topic_id],
                       label=f'主题 {topic_id+1}')
        
        ax5.set_xlabel('台词序列')
        ax5.set_ylabel('情感强度')
        ax5.set_title('主题-情感关系')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. 高频词条形图
        ax6 = plt.subplot(3, 3, 6)
        if vocab_stats['top_content_words']:
            top_words = list(vocab_stats['top_content_words'].items())[:15]
            words, freqs = zip(*top_words)
            y_pos = np.arange(len(words))
            
            ax6.barh(y_pos, freqs, color='salmon', alpha=0.7)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(words, fontsize=9)
            ax6.set_xlabel('出现频率')
            ax6.set_title('高频词（排除停用词）')
            ax6.invert_yaxis()
            ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. 主题情感箱线图
        ax7 = plt.subplot(3, 3, 7)
        topic_data = []
        topic_labels = []
        
        for topic_id in range(num_topics):
            topic_data.append(df[df['dominant_topic'] == topic_id]['compound'].values)
            topic_labels.append(f'主题 {topic_id+1}')
        
        if all(len(data) > 0 for data in topic_data):
            bp = ax7.boxplot(topic_data, labels=topic_labels, patch_artist=True)
            
            # 设置箱线图颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax7.set_ylabel('情感强度')
        ax7.set_title('各主题情感分布')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. 主题置信度分布
        ax8 = plt.subplot(3, 3, 8)
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            if topic_mask.sum() > 0:
                ax8.hist(df.loc[topic_mask, 'topic_confidence'], 
                        bins=20, alpha=0.5, color=colors[topic_id],
                        label=f'主题 {topic_id+1}')
        
        ax8.set_xlabel('主题置信度')
        ax8.set_ylabel('频率')
        ax8.set_title('主题置信度分布')
        if num_topics <= 6:
            ax8.legend(loc='upper right', fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        # 9. 情感分类饼图
        ax9 = plt.subplot(3, 3, 9)
        sentiment_counts = df['sentiment'].value_counts()
        colors_pie = ['lightgreen', 'lightcoral', 'lightblue']
        wedges, texts, autotexts = ax9.pie(sentiment_counts.values, 
                                          labels=sentiment_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors_pie,
                                          startangle=90)
        
        # 美化百分比文本
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax9.set_title('情感分类比例')
        
        plt.suptitle('动漫台词文本分析仪表板', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图形
        plt.savefig('anime_dialogue_analysis.png', dpi=150, bbox_inches='tight')
        print("可视化图表已保存为 'anime_dialogue_analysis.png'")
        
        plt.show()
        
        # 如果有角色数据，创建额外图表
        character_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['character', 'role', 'speaker', '角色', '说话者'])]
        
        if character_cols:
            character_col = character_cols[0]
            if df[character_col].dtype == 'object' and df[character_col].nunique() < 20:
                plt.figure(figsize=(12, 6))
                character_sentiment = df.groupby(character_col)['compound'].mean().sort_values()
                
                # 角色情感条形图
                colors_char = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(character_sentiment)))
                bars = plt.barh(range(len(character_sentiment)), character_sentiment.values, color=colors_char)
                
                plt.xlabel('平均情感强度')
                plt.ylabel('角色')
                plt.yticks(range(len(character_sentiment)), character_sentiment.index)
                plt.title('角色情感分析')
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3, axis='x')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars, character_sentiment.values)):
                    color = 'black' if abs(val) < 0.5 else 'white'
                    plt.text(val + (0.01 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', color=color, va='center',
                            fontweight='bold' if abs(val) > 0.3 else 'normal')
                
                plt.tight_layout()
                plt.savefig('character_sentiment_analysis.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("角色情感分析图表已保存为 'character_sentiment_analysis.png'")
        
        return fig
        
    except Exception as e:
        print(f"创建可视化图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# 第七部分：高级分析（聚类分析）
# ============================================

def perform_clustering_analysis(df, topic_distribution):
    """
    执行聚类分析
    """
    print("\n" + "="*60)
    print("开始聚类分析...")
    
    # 使用主题分布进行聚类
    X_cluster = topic_distribution
    
    # 寻找最佳聚类数量
    silhouette_scores = []
    k_range = range(2, min(8, len(df) // 20 + 2))
    
    if len(k_range) > 1 and len(df) > 20:
        print("寻找最佳聚类数量...")
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_cluster)
                silhouette_avg = silhouette_score(X_cluster, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"  k={k}, Silhouette Score: {silhouette_avg:.4f}")
            except:
                silhouette_scores.append(-1)
        
        # 选择最佳k值
        if max(silhouette_scores) > 0:
            best_k = k_range[np.argmax(silhouette_scores)]
        else:
            best_k = 3
    else:
        best_k = 3
    
    print(f"最佳聚类数量: {best_k}")
    
    # 执行聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster)
    
    # 分析每个聚类的特征
    print("\n聚类分析结果:")
    cluster_summary = []
    for cluster_id in range(best_k):
        cluster_mask = df['cluster'] == cluster_id
        cluster_size = cluster_mask.sum()
        
        if cluster_size == 0:
            continue
        
        # 聚类特征
        avg_sentiment = df.loc[cluster_mask, 'compound'].mean()
        
        # 找到主要主题
        if cluster_size > 0:
            topic_counts = df.loc[cluster_mask, 'dominant_topic'].value_counts()
            if len(topic_counts) > 0:
                dominant_topic = topic_counts.idxmax()
                topic_pct = topic_counts.max() / cluster_size
            else:
                dominant_topic = -1
                topic_pct = 0
        else:
            dominant_topic = -1
            topic_pct = 0
        
        # 找到代表性台词（情感最强烈的）
        if cluster_size > 0:
            try:
                representative = df[cluster_mask].loc[df[cluster_mask]['compound'].abs().idxmax()]
                example_line = representative['cleaned_line']
                if len(example_line) > 80:
                    example_line = example_line[:80] + "..."
            except:
                example_line = "无"
        else:
            example_line = "无"
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'size_pct': cluster_size / len(df),
            'avg_sentiment': avg_sentiment,
            'dominant_topic': dominant_topic,
            'topic_pct': topic_pct,
            'example': example_line
        })
        
        print(f"\n聚类 {cluster_id}:")
        print(f"  大小: {cluster_size} ({cluster_size/len(df):.1%})")
        print(f"  平均情感: {avg_sentiment:.3f}")
        if dominant_topic != -1:
            print(f"  主要主题: 主题 {dominant_topic + 1} ({topic_pct:.1%})")
        print(f"  代表性台词: {example_line}")
    
    # 可视化聚类
    if len(df) > 30 and best_k > 1:
        try:
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 4))
            X_tsne = tsne.fit_transform(X_cluster)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=df['cluster'], cmap='tab10',
                                 alpha=0.6, s=30)
            plt.colorbar(scatter, label='聚类')
            plt.xlabel('t-SNE 维度 1')
            plt.ylabel('t-SNE 维度 2')
            plt.title(f'台词聚类可视化 (k={best_k})')
            plt.grid(True, alpha=0.3)
            plt.savefig('clustering_visualization.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("聚类可视化已保存为 'clustering_visualization.png'")
        except Exception as e:
            print(f"聚类可视化失败: {e}")
    
    return df, cluster_summary


# ============================================
# 第八部分：生成分析报告
# ============================================

def generate_analysis_report(df, topics_dict, vocab_stats, turning_points, cluster_summary):
    """
    生成详细的分析报告
    """
    print("\n" + "="*60)
    print("生成分析报告...")
    
    # 计算报告所需的各种统计数据
    length_stats = {
        'avg_chars': df['char_length'].mean(),
        'std_chars': df['char_length'].std(),
        'avg_words': df['word_count'].mean(),
        'std_words': df['word_count'].std(),
        'max_chars': df['char_length'].max(),
        'min_chars': df['char_length'].min(),
        'max_words': df['word_count'].max(),
        'min_words': df['word_count'].min()
    }
    
    sentiment_stats = {
        'positive': (df['sentiment'] == 'positive').sum(),
        'negative': (df['sentiment'] == 'negative').sum(),
        'neutral': (df['sentiment'] == 'neutral').sum(),
        'positive_pct': (df['sentiment'] == 'positive').sum() / len(df),
        'negative_pct': (df['sentiment'] == 'negative').sum() / len(df),
        'neutral_pct': (df['sentiment'] == 'neutral').sum() / len(df),
        'avg_compound': df['compound'].mean(),
        'avg_compound_abs': df['compound'].abs().mean()
    }
    
    # 获取元数据信息
    metadata_info = ""
    metadata_cols = [col for col in df.columns if col not in 
                    ['raw_line', 'line_number', 'cleaned_line', 'char_length', 
                     'word_count', 'sentiment', 'compound', 'positive', 'negative', 
                     'neutral', 'compound_ma', 'processed_text', 'dominant_topic',
                     'topic_confidence', 'cluster']]
    
    if metadata_cols:
        metadata_info = "\n\n数据集元数据:\n"
        for col in metadata_cols:
            if col in df.columns:
                metadata_info += f"  • {col}: {df[col].dtype}, {df[col].nunique()}个唯一值\n"
                if df[col].dtype == 'object' and df[col].nunique() < 10:
                    unique_vals = df[col].unique()
                    metadata_info += f"    值: {', '.join(map(str, unique_vals))}\n"
    
    # 生成报告文本
    report = f"""
{'='*80}
动漫台词文本数据分析报告
{'='*80}

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

一、数据集概览
{'─'*40}
• 总台词数: {len(df):,} 句
• 总词数: {vocab_stats['total_words']:,} 个
• 独特词汇: {vocab_stats['unique_words']:,} 个
• 词汇多样性: {vocab_stats['lexical_diversity']:.2%}
{metadata_info}

二、台词长度分析
{'─'*40}
• 平均字符长度: {length_stats['avg_chars']:.1f} (±{length_stats['std_chars']:.1f})
• 平均词数: {length_stats['avg_words']:.1f} (±{length_stats['std_words']:.1f})
• 最长台词: {length_stats['max_chars']} 字符 ({length_stats['max_words']} 词)
• 最短台词: {length_stats['min_chars']} 字符 ({length_stats['min_words']} 词)

三、情感分析结果
{'─'*40}
• 积极台词: {sentiment_stats['positive']:,} 句 ({sentiment_stats['positive_pct']:.1%})
• 消极台词: {sentiment_stats['negative']:,} 句 ({sentiment_stats['negative_pct']:.1%})
• 中性台词: {sentiment_stats['neutral']:,} 句 ({sentiment_stats['neutral_pct']:.1%})
• 平均情感强度: {sentiment_stats['avg_compound_abs']:.3f}
• 整体情感基调: {"积极向上" if sentiment_stats['avg_compound'] > 0.1 else "消极低沉" if sentiment_stats['avg_compound'] < -0.1 else "中性平衡"}
• 情感转折点: {len(turning_points)} 个

四、主题建模发现
{'─'*40}
共发现 {len(topics_dict)} 个主题:

"""
    
    # 添加主题详情
    for topic_name, words in topics_dict.items():
        topic_id = int(topic_name.split('_')[1]) - 1
        topic_count = (df['dominant_topic'] == topic_id).sum()
        topic_pct = topic_count / len(df)
        
        # 计算该主题的平均情感
        topic_sentiment = df[df['dominant_topic'] == topic_id]['compound'].mean()
        sentiment_label = "积极" if topic_sentiment > 0.1 else "消极" if topic_sentiment < -0.1 else "中性"
        
        report += f"• {topic_name}: {', '.join(words[:8])}\n"
        report += f"  数量: {topic_count}句 ({topic_pct:.1%}) | "
        report += f"平均情感: {topic_sentiment:.3f} ({sentiment_label})\n\n"
    
    # 添加聚类分析结果
    if cluster_summary:
        report += f"五、聚类分析结果\n{'─'*40}\n"
        report += f"共发现 {len(cluster_summary)} 个聚类:\n\n"
        
        for cluster in cluster_summary:
            report += f"• 聚类 {cluster['cluster_id']}:\n"
            report += f"  大小: {cluster['size']}句 ({cluster['size_pct']:.1%})\n"
            report += f"  平均情感: {cluster['avg_sentiment']:.3f}\n"
            if cluster['dominant_topic'] != -1:
                report += f"  主要主题: 主题 {cluster['dominant_topic'] + 1} ({cluster['topic_pct']:.1%})\n"
            report += f"  代表性台词: {cluster['example']}\n\n"
    
    # 添加高频词汇
    report += f"六、高频词汇分析\n{'─'*40}\n"
    report += "前15个高频内容词（排除停用词）:\n\n"
    
    if vocab_stats['top_content_words']:
        for i, (word, freq) in enumerate(list(vocab_stats['top_content_words'].items())[:15], 1):
            freq_pct = freq / vocab_stats['total_words'] if vocab_stats['total_words'] > 0 else 0
            report += f"{i:2d}. {word:15s}: {freq:5d}次 ({freq_pct:.2%})\n"
    else:
        report += "无足够的高频词数据\n"
    
    # 添加叙事洞察
    report += f"\n七、叙事洞察\n{'─'*40}\n"
    
    # 分析情感变化模式
    if len(turning_points) > 0:
        report += "• 情感叙事有明显起伏，共发现 {} 个情感转折点\n".format(len(turning_points))
        
        # 分析前半段和后半段的情感
        if len(df) > 10:
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]['compound'].mean()
            second_half = df.iloc[mid_point:]['compound'].mean()
            
            if abs(second_half - first_half) > 0.2:
                if second_half > first_half:
                    report += "• 叙事后半段比前半段更积极，情感呈上升趋势\n"
                else:
                    report += "• 叙事后半段比前半段更消极，情感呈下降趋势\n"
    
    # 分析主题分布模式
    if 'dominant_topic' in df.columns and len(df) > 1:
        topic_sequence = df['dominant_topic'].values
        topic_changes = sum(1 for i in range(1, len(topic_sequence)) 
                           if topic_sequence[i] != topic_sequence[i-1])
        
        avg_topic_length = len(df) / (topic_changes + 1) if topic_changes > 0 else len(df)
        report += f"• 平均每 {avg_topic_length:.0f} 句台词发生一次主题转换\n"
        
        # 分析最长的主题连续段
        if len(topic_sequence) > 0:
            current_topic = topic_sequence[0]
            current_length = 1
            max_length = 1
            max_topic = current_topic
            
            for i in range(1, len(topic_sequence)):
                if topic_sequence[i] == current_topic:
                    current_length += 1
                    if current_length > max_length:
                        max_length = current_length
                        max_topic = current_topic
                else:
                    current_topic = topic_sequence[i]
                    current_length = 1
            
            report += f"• 最长的主题连续段: 主题 {max_topic + 1}，连续 {max_length} 句台词\n"
    
    # 添加情感转折点详情
    if turning_points:
        report += f"\n八、主要情感转折点\n{'─'*40}\n"
        for i, (point_type, idx, value, line_text) in enumerate(turning_points[:10], 1):
            point_name = "高潮" if point_type == 'peak' else "低谷"
            report += f"{i}. 第{idx+1}句: {point_name} (情感强度: {value:.3f})\n"
            report += f"   台词: {line_text}...\n\n"
    
    # 添加分析建议
    report += f"\n九、分析建议\n{'─'*40}\n"
    report += "1. 结合角色识别: 使用命名实体识别自动识别说话者\n"
    report += "2. 上下文分析: 考虑前后台词的情感传递关系\n"
    report += "3. 文化语境: 添加更多日语动漫特有词汇到情感词典\n"
    report += "4. 多作品比较: 分析不同导演或时期的台词风格差异\n"
    report += "5. 叙事结构: 结合经典叙事理论分析起承转合\n"
    
    # 添加技术说明
    report += f"\n十、技术说明\n{'─'*40}\n"
    report += "• 情感分析: 使用NLTK的VADER，扩展了动漫相关词汇\n"
    report += "• 主题建模: 使用LDA算法，TF-IDF向量化\n"
    report += "• 聚类分析: 使用K-Means算法，轮廓系数选择最佳K值\n"
    report += "• 可视化: 使用Matplotlib和Seaborn创建综合仪表板\n"
    report += "• 文件支持: 支持.txt, .csv, .xlsx, .xls格式\n"
    
    report += f"\n{'='*80}\n分析完成！\n{'='*80}"
    
    # 保存报告
    try:
        with open('anime_dialogue_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("分析报告已保存为 'anime_dialogue_analysis_report.txt'")
    except Exception as e:
        print(f"保存报告文件失败: {e}")
    
    # 在控制台显示报告摘要
    print("\n" + "="*60)
    print("报告摘要:")
    print(f"• 分析台词数: {len(df):,}")
    print(f"• 情感分布: {sentiment_stats['positive_pct']:.1%} 积极, "
          f"{sentiment_stats['negative_pct']:.1%} 消极, "
          f"{sentiment_stats['neutral_pct']:.1%} 中性")
    print(f"• 发现主题数: {len(topics_dict)}")
    if cluster_summary:
        print(f"• 发现聚类数: {len(cluster_summary)}")
    print(f"• 情感转折点: {len(turning_points)} 个")
    print("="*60)
    
    return report


# ============================================
# 第九部分：导出结果
# ============================================

def export_results(df, topics_dict):
    """
    导出分析结果
    """
    print("\n" + "="*60)
    print("导出分析结果...")
    
    # 导出清洗后的数据
    export_df = df.copy()
    
    # 添加主题关键词
    topic_keywords_dict = {}
    for topic_name, words in topics_dict.items():
        topic_id = int(topic_name.split('_')[1]) - 1
        topic_keywords_dict[topic_id] = ', '.join(words[:10])
    
    export_df['topic_keywords'] = export_df['dominant_topic'].map(topic_keywords_dict)
    
    # 选择要导出的列（保留原始元数据）
    export_columns = [
        'line_number', 
        'cleaned_line', 
        'char_length', 
        'word_count',
        'sentiment', 
        'compound',
        'dominant_topic',
        'topic_keywords',
        'topic_confidence'
    ]
    
    if 'cluster' in export_df.columns:
        export_columns.append('cluster')
    
    # 添加其他元数据列（如果有）
    base_columns = ['raw_line', 'line_number', 'cleaned_line', 'char_length', 
                   'word_count', 'sentiment', 'compound', 'positive', 'negative', 
                   'neutral', 'compound_ma', 'processed_text', 'dominant_topic',
                   'topic_confidence', 'topic_keywords', 'cluster']
    
    extra_cols = [col for col in export_df.columns if col not in base_columns]
    export_columns.extend(extra_cols)
    
    # 确保所有列都存在
    export_columns = [col for col in export_columns if col in export_df.columns]
    
    export_df = export_df[export_columns]
    
    # 保存为CSV
    csv_path = 'anime_dialogue_analysis_results.csv'
    try:
        export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"分析结果已保存为 '{csv_path}'")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")
    
    # 导出主题详情
    try:
        topics_df = pd.DataFrame(topics_dict)
        topics_path = 'anime_dialogue_topics.csv'
        topics_df.to_csv(topics_path, encoding='utf-8-sig')
        print(f"主题详情已保存为 '{topics_path}'")
    except Exception as e:
        print(f"保存主题详情失败: {e}")
    
    return export_df


# ============================================
# 第十部分：主函数
# ============================================

def main():
    """
    主函数：执行完整分析流程
    """
    print("="*60)
    print("动漫台词文本数据分析系统")
    print("="*60)
    
    # 【关键修复】：修改为支持Excel文件
    file_path = r"D:\python\红猪\副本红猪.xlsx"  # 您的Excel文件路径
    
    # 如果您想分析文本文件，可以修改为：
    # file_path = "anime_dialogue.txt"
    
    # Excel文件中台词文本所在的列名（如果您的Excel列名不同，请修改这里）
    # 常见列名：'dialogue', 'text', '台词', '对话', 'line', 'script' 等
    text_column = "cueline"  # 如果您的Excel中台词列不是'dialogue'，请修改
    
    try:
        # 步骤1: 加载和清洗数据
        print("\n步骤1: 加载和清洗数据...")
        df = load_and_clean_data(file_path, text_column)
        
        if df is None:
            print("数据加载失败，请检查文件路径和格式")
            return
        
        if len(df) < 10:
            print("警告: 数据量较少，分析结果可能不准确")
            print("建议至少提供50句台词以获得更好的分析结果")
            # 可以选择继续或退出
            # return
        
        print(f"成功加载 {len(df)} 句台词")
        
        # 步骤2: 词汇统计分析
        vocab_stats = analyze_vocabulary(df)
        
        # 步骤3: 情感分析
        df, turning_points = perform_sentiment_analysis(df)
        
        # 步骤4: 主题建模
        df, topics_dict, lda_model, X_matrix, vectorizer = perform_topic_modeling(df)
        
        # 步骤5: 可视化分析
        fig = create_visualizations(df, topics_dict, turning_points, vocab_stats)
        
        # 步骤6: 聚类分析（可选，数据量足够时执行）
        if len(df) > 30:
            topic_distribution = lda_model.transform(
                vectorizer.transform(df['processed_text'])
            )
            df, cluster_summary = perform_clustering_analysis(df, topic_distribution)
        else:
            print("\n数据量较少，跳过聚类分析")
            cluster_summary = []
        
        # 步骤7: 生成分析报告
        report = generate_analysis_report(
            df, topics_dict, vocab_stats, turning_points, cluster_summary
        )
        
        # 步骤8: 导出结果
        export_df = export_results(df, topics_dict)
        
        print("\n" + "="*60)
        print("分析完成！")
        print("生成的文件:")
        print("  1. anime_dialogue_analysis.png - 可视化图表")
        print("  2. anime_dialogue_analysis_report.txt - 详细分析报告")
        print("  3. anime_dialogue_analysis_results.csv - 分析结果数据")
        print("  4. anime_dialogue_topics.csv - 主题详情")
        if len(df) > 30 and cluster_summary:
            print("  5. clustering_visualization.png - 聚类可视化")
        print("="*60)
        
        # 显示一些示例分析结果
        print("\n示例分析结果:")
        if len(df) > 0:
            # 最长台词
            longest_idx = df['char_length'].idxmax()
            print(f"1. 最长台词 (第{longest_idx+1}句):")
            print(f"   '{df.loc[longest_idx, 'cleaned_line'][:100]}...'")
            
            # 最积极台词
            if df['compound'].max() > 0:
                most_positive_idx = df['compound'].idxmax()
                print(f"\n2. 最积极台词 (第{most_positive_idx+1}句):")
                print(f"   '{df.loc[most_positive_idx, 'cleaned_line'][:100]}...'")
            
            # 最消极台词
            if df['compound'].min() < 0:
                most_negative_idx = df['compound'].idxmin()
                print(f"\n3. 最消极台词 (第{most_negative_idx+1}句):")
                print(f"   '{df.loc[most_negative_idx, 'cleaned_line'][:100]}...'")
            
            # 情感最强烈的台词
            if df['compound'].abs().max() > 0:
                most_intense_idx = df['compound'].abs().idxmax()
                sentiment = "积极" if df.loc[most_intense_idx, 'compound'] > 0 else "消极"
                print(f"\n4. 情感最强烈的台词 (第{most_intense_idx+1}句):")
                print(f"   {sentiment}情绪: '{df.loc[most_intense_idx, 'cleaned_line'][:100]}...'")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        print("请确保文件存在，或修改文件路径")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================
# 执行主函数
# ============================================

if __name__ == "__main__":
    # 安装必要的库（如果未安装）
    print("检查依赖库...")
    required_libraries = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'nltk', 'scikit-learn']
    
    for lib in required_libraries:
        try:
            __import__(lib)
            print(f"  ✓ {lib} 已安装")
        except ImportError:
            print(f"  ✗ {lib} 未安装，请运行: pip install {lib}")
    
    # 特别检查openpyxl（用于Excel支持）
    try:
        import openpyxl
        print("  ✓ openpyxl 已安装 (Excel支持)")
    except ImportError:
        print("  ✗ openpyxl 未安装，如果需要Excel支持，请运行: pip install openpyxl")
    
    print("\n" + "="*60)
    
    # 运行完整分析流程
    main()
    
    print("\n使用说明:")
    print("1. 准备您的数据文件:")
    print("   - 文本文件 (.txt): 每行一句台词")
    print("   - Excel文件 (.xlsx/.xls): 包含台词列的表格")
    print("   - CSV文件 (.csv): 包含台词列的表格")
    print("2. 修改main()函数中的file_path变量为您的文件路径")
    print("3. 如果需要，修改text_column变量为您的台词列名")
    print("4. 运行此脚本")
    print("5. 查看生成的分析文件和报告")

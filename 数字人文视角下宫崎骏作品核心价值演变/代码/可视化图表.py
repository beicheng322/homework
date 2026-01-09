# ============================================
# 动漫台词文本数据分析完整流程（修复乱码版）
# 版本：2.2
# 功能：支持Excel和文本文件，解决图表乱码问题
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

# ============================================
# 修复字体问题，避免中文乱码
# ============================================

# 方法1：使用英文标签（最简单，不会乱码）
USE_ENGLISH_LABELS = True  # 设置为True使用英文标签，False使用中文标签

# 方法2：如果必须使用中文，尝试设置正确的字体
if not USE_ENGLISH_LABELS:
    import matplotlib
    # 尝试不同的中文字体解决方案
    try:
        # Windows系统常用中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        print("已设置中文字体")
    except:
        print("中文字体设置失败，将使用英文标签")
        USE_ENGLISH_LABELS = True

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("所有库导入完成！\n" + "="*60)


# ============================================
# 定义中英文标签对照表
# ============================================

if USE_ENGLISH_LABELS:
    # 英文标签
    LABELS = {
        # 通用标签
        'dialogue_sequence': 'Dialogue Sequence',
        'sentiment_intensity': 'Sentiment Intensity',
        'frequency': 'Frequency',
        'word_count': 'Word Count',
        'topic': 'Topic',
        'topic_confidence': 'Topic Confidence',
        'average': 'Average',
        'topic_distribution': 'Topic Distribution',
        'sentiment_distribution': 'Sentiment Distribution',
        'dialogue_length': 'Dialogue Length',
        'high_freq_words': 'High Frequency Words',
        'character': 'Character',
        'sentiment_analysis': 'Sentiment Analysis',
        'clustering': 'Clustering',
        
        # 图表标题
        'sentiment_narrative': 'Sentiment Narrative Curve',
        'sentiment_histogram': 'Sentiment Intensity Distribution',
        'length_distribution': 'Dialogue Length Distribution',
        'topic_over_time': 'Topic Distribution Over Time',
        'topic_sentiment': 'Topic-Sentiment Relationship',
        'word_frequency': 'High Frequency Words (Stopwords Removed)',
        'topic_boxplot': 'Sentiment Distribution by Topic',
        'confidence_distribution': 'Topic Confidence Distribution',
        'sentiment_pie': 'Sentiment Classification Ratio',
        
        # 情感分类
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        
        # 说明文本
        'explanation_sentiment': 'Explanation:\n• Blue line: Original sentiment per dialogue\n• Red line: Smoothed sentiment trend\n• Green triangle: Emotional peak\n• Red inverted triangle: Emotional valley',
        'explanation_histogram': 'Statistics:\n• Positive dialogues: {pos_count} ({pos_pct:.1f}%)\n• Negative dialogues: {neg_count} ({neg_pct:.1f}%)\n• Neutral dialogues: {neu_count} ({neu_pct:.1f}%)',
        'explanation_length': 'Length Statistics:\n• Avg characters: {avg_chars:.1f}\n• Char range: {min_chars}-{max_chars}\n• Word range: {min_words}-{max_words}',
        'explanation_vocab': 'Vocabulary Stats:\n• Total words: {total_words:,}\n• Unique words: {unique_words:,}\n• Lexical diversity: {lexical_diversity:.2%}',
        'explanation_topic': 'Topic Keywords:\n{topic_keywords}',
        'explanation_boxplot': 'Boxplot Explanation:\n• Box: 25%-75% data range\n• Middle line: Median\n• White dot: Mean\n• Whiskers: Data range',
        
        # 其他
        'peak': 'Peak',
        'valley': 'Valley',
        'emotion': 'Emotion',
        'strength': 'Strength',
        'lines': 'lines',
        'characters': 'characters',
        'words': 'words',
        'density': 'Density',
        'count': 'Count',
        'topic_id': 'Topic {id}',
        'cluster_id': 'Cluster {id}',
        'dominant_topic': 'Dominant Topic',
        'main_topic': 'Main Topic',
        'representative_dialogue': 'Representative Dialogue',
        'example': 'Example',
        'dashboard_title': 'Anime Dialogue Text Analysis Dashboard',
    }
else:
    # 中文标签
    LABELS = {
        # 通用标签
        'dialogue_sequence': '台词序列号',
        'sentiment_intensity': '情感强度',
        'frequency': '频率',
        'word_count': '台词词数',
        'topic': '主题',
        'topic_confidence': '主题置信度',
        'average': '平均',
        'topic_distribution': '主题分布',
        'sentiment_distribution': '情感分布',
        'dialogue_length': '台词长度',
        'high_freq_words': '高频词汇',
        'character': '角色',
        'sentiment_analysis': '情感分析',
        'clustering': '聚类分析',
        
        # 图表标题
        'sentiment_narrative': '情感叙事曲线分析',
        'sentiment_histogram': '情感强度分布直方图',
        'length_distribution': '台词长度分布（按词数）',
        'topic_over_time': '主题随时间分布',
        'topic_sentiment': '主题-情感关系散点图',
        'word_frequency': '高频词分析（排除停用词）',
        'topic_boxplot': '各主题情感分布箱线图',
        'confidence_distribution': '主题置信度分布',
        'sentiment_pie': '情感分类比例饼图',
        
        # 情感分类
        'positive': '积极',
        'negative': '消极',
        'neutral': '中性',
        
        # 说明文本
        'explanation_sentiment': '说明：\n• 蓝色线：每句台词原始情感分数\n• 红色线：移动平均平滑后情感趋势\n• 绿三角：情感高潮点\n• 红倒三角：情感低谷点',
        'explanation_histogram': '统计信息：\n• 积极台词: {pos_count}句 ({pos_pct:.1f}%)\n• 消极台词: {neg_count}句 ({neg_pct:.1f}%)\n• 中性台词: {neu_count}句 ({neu_pct:.1f}%)',
        'explanation_length': '长度统计：\n• 平均字符数: {avg_chars:.1f}\n• 字符数范围: {min_chars}-{max_chars}\n• 词数范围: {min_words}-{max_words}',
        'explanation_vocab': '词汇统计：\n• 总词数: {total_words:,}\n• 独特词汇: {unique_words:,}\n• 词汇多样性: {lexical_diversity:.2%}',
        'explanation_topic': '主题关键词：\n{topic_keywords}',
        'explanation_boxplot': '箱线图说明：\n• 箱子: 25%-75%数据范围\n• 中线: 中位数\n• 白点: 平均值\n• 须线: 数据范围',
        
        # 其他
        'peak': '高潮',
        'valley': '低谷',
        'emotion': '情感',
        'strength': '强度',
        'lines': '句',
        'characters': '字符',
        'words': '词',
        'density': '密度',
        'count': '数量',
        'topic_id': '主题 {id}',
        'cluster_id': '聚类 {id}',
        'dominant_topic': '主要主题',
        'main_topic': '主题',
        'representative_dialogue': '代表性台词',
        'example': '示例',
        'dashboard_title': '动漫台词文本分析仪表板',
    }


# ============================================
# 第二部分：数据加载与预处理（支持Excel和文本）
# ============================================

def load_and_clean_data(file_path, text_column='dialogue'):
    """
    加载和清洗台词数据（支持Excel和文本文件）
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
                          ['text', 'dialogue', 'line', '台词', '对话', '脚本']):
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
        line = re.sub(r'\[.*?]', '', line)      # 移除方括号内容
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
# 第六部分：可视化分析（修复乱码版）
# ============================================

def create_visualizations(df, topics_dict, turning_points, vocab_stats):
    """
    创建所有可视化图表（修复乱码版）
    """
    print("\n" + "="*60)
    print("创建可视化图表...")
    
    try:
        # 创建图形
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 情感叙事曲线
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(df.index, df['compound'], alpha=0.5, linewidth=0.8, label='Original', color='lightblue')
        
        if 'compound_ma' in df.columns and df['compound_ma'].notna().sum() > 0:
            ax1.plot(df.index, df['compound_ma'], linewidth=2, label='Smoothed', color='red')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
        
        # 标记转折点
        for point_type, idx, value, line_text in turning_points[:5]:  # 只标记前5个
            color = 'green' if point_type == 'peak' else 'red'
            marker = '^' if point_type == 'peak' else 'v'
            label = 'Peak' if point_type == 'peak' else 'Valley'
            ax1.scatter(idx, value, color=color, marker=marker, s=100, zorder=5, label=label)
        
        ax1.set_xlabel(LABELS['dialogue_sequence'], fontsize=10)
        ax1.set_ylabel(LABELS['sentiment_intensity'], fontsize=10)
        ax1.set_title(LABELS['sentiment_narrative'], fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 添加图表说明
        if not USE_ENGLISH_LABELS:
            ax1.text(0.02, 0.02, LABELS['explanation_sentiment'],
                    transform=ax1.transAxes, fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. 情感分布直方图
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(df['compound'], bins=30, edgecolor='black', alpha=0.7, color='skyblue', orientation='horizontal')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        
        # 标注平均情感
        mean_sentiment = df['compound'].mean()
        ax2.axhline(y=mean_sentiment, color='green', linestyle='-', alpha=0.8, label=f'Avg: {mean_sentiment:.3f}')
        
        ax2.set_xlabel(LABELS['frequency'], fontsize=10)
        ax2.set_ylabel(LABELS['sentiment_intensity'], fontsize=10)
        ax2.set_title(LABELS['sentiment_histogram'], fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息注释
        positive_count = (df['compound'] > 0.05).sum()
        negative_count = (df['compound'] < -0.05).sum()
        neutral_count = len(df) - positive_count - negative_count
        total_count = len(df)
        
        pos_pct = positive_count/total_count*100
        neg_pct = negative_count/total_count*100
        neu_pct = neutral_count/total_count*100
        
        stats_text = f"Statistics:\n• Positive: {positive_count} ({pos_pct:.1f}%)\n• Negative: {negative_count} ({neg_pct:.1f}%)\n• Neutral: {neutral_count} ({neu_pct:.1f}%)"
        
        ax2.text(0.02, 0.02, stats_text,
                transform=ax2.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 3. 台词长度分布
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(df['word_count'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax3.axvline(df['word_count'].mean(), color='red', linestyle='--', 
                    label=f'{LABELS["average"]}: {df["word_count"].mean():.1f}')
        
        ax3.set_xlabel(LABELS['word_count'], fontsize=10)
        ax3.set_ylabel(LABELS['frequency'], fontsize=10)
        ax3.set_title(LABELS['length_distribution'], fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_chars = df['char_length'].mean()
        min_chars = df['char_length'].min()
        max_chars = df['char_length'].max()
        min_words = df['word_count'].min()
        max_words = df['word_count'].max()
        
        length_stats = f"Length Stats:\n• Avg chars: {avg_chars:.1f}\n• Char range: {min_chars}-{max_chars}\n• Word range: {min_words}-{max_words}"
        
        ax3.text(0.02, 0.02, length_stats,
                transform=ax3.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 4. 主题随时间分布
        ax4 = plt.subplot(3, 3, 4)
        num_topics = len(topics_dict)
        colors = plt.cm.Set3(np.linspace(0, 1, num_topics))
        
        topic_labels = []
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            if topic_mask.sum() > 0:
                ax4.scatter(df[topic_mask].index, 
                           [topic_id + 1] * len(df[topic_mask]),
                           alpha=0.6, s=15, color=colors[topic_id],
                           label=f'Topic {topic_id+1}')
                topic_labels.append(f'Topic {topic_id+1}')
        
        ax4.set_xlabel(LABELS['dialogue_sequence'], fontsize=10)
        ax4.set_ylabel(LABELS['topic'], fontsize=10)
        ax4.set_title(LABELS['topic_over_time'], fontsize=12, fontweight='bold')
        ax4.set_yticks(range(1, num_topics + 1))
        ax4.set_yticklabels([f'Topic {i+1}' for i in range(num_topics)])
        
        if num_topics <= 8:
            ax4.legend(loc='upper right', fontsize=8, ncol=2)
        
        ax4.grid(True, alpha=0.3)
        
        # 添加主题关键词注释
        topic_keywords_text = "Topic Keywords:\n"
        for i, (topic_name, words) in enumerate(list(topics_dict.items())[:min(4, len(topics_dict))]):
            topic_keywords_text += f"• {topic_name}: {', '.join(words[:3])}\n"
        
        ax4.text(0.02, 0.98, topic_keywords_text,
                transform=ax4.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 5. 主题-情感关系
        ax5 = plt.subplot(3, 3, 5)
        
        # 计算每个主题的平均情感
        topic_avg_sentiment = []
        topic_names = []
        
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            if topic_mask.sum() > 0:
                avg_sent = df[topic_mask]['compound'].mean()
                topic_avg_sentiment.append(avg_sent)
                topic_names.append(f'Topic {topic_id+1}')
                
                ax5.scatter(df[topic_mask].index, 
                           df[topic_mask]['compound'],
                           alpha=0.5, s=15, color=colors[topic_id],
                           label=f'Topic {topic_id+1}')
        
        ax5.set_xlabel(LABELS['dialogue_sequence'], fontsize=10)
        ax5.set_ylabel(LABELS['sentiment_intensity'], fontsize=10)
        ax5.set_title(LABELS['topic_sentiment'], fontsize=12, fontweight='bold')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if num_topics <= 6:
            ax5.legend(loc='upper right', fontsize=8, ncol=2)
        
        ax5.grid(True, alpha=0.3)
        
        # 添加主题情感统计
        if topic_avg_sentiment:
            sentiment_summary = "Topic Sentiment:\n"
            for i, (name, sentiment) in enumerate(zip(topic_names, topic_avg_sentiment)):
                sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                sentiment_summary += f"• {name}: {sentiment:.3f} ({sentiment_label})\n"
            
            ax5.text(0.02, 0.98, sentiment_summary,
                    transform=ax5.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 6. 高频词条形图
        ax6 = plt.subplot(3, 3, 6)
        if vocab_stats['top_content_words']:
            top_words = list(vocab_stats['top_content_words'].items())[:15]
            words, freqs = zip(*top_words)
            y_pos = np.arange(len(words))
            
            bars = ax6.barh(y_pos, freqs, color='salmon', alpha=0.7, edgecolor='darkred')
            
            # 为每个条形添加数值标签
            for bar, freq in zip(bars, freqs):
                width = bar.get_width()
                ax6.text(width + max(freqs)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{freq}', va='center', fontsize=8)
            
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(words, fontsize=9)
            ax6.set_xlabel(LABELS['frequency'], fontsize=10)
            ax6.set_title(LABELS['word_frequency'], fontsize=12, fontweight='bold')
            ax6.invert_yaxis()
            ax6.grid(True, alpha=0.3, axis='x')
            
            # 添加词汇统计注释
            total_words = vocab_stats['total_words']
            unique_words = vocab_stats['unique_words']
            lexical_diversity = vocab_stats['lexical_diversity']
            
            vocab_info = f"Vocabulary Stats:\n• Total words: {total_words:,}\n• Unique words: {unique_words:,}\n• Lexical diversity: {lexical_diversity:.2%}"
            
            ax6.text(0.02, 0.02, vocab_info,
                    transform=ax6.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 7. 主题情感箱线图
        ax7 = plt.subplot(3, 3, 7)
        topic_data = []
        topic_labels_box = []
        
        for topic_id in range(num_topics):
            topic_data.append(df[df['dominant_topic'] == topic_id]['compound'].values)
            topic_labels_box.append(f'Topic {topic_id+1}')
        
        if all(len(data) > 0 for data in topic_data):
            bp = ax7.boxplot(topic_data, labels=topic_labels_box, patch_artist=True, showmeans=True)
            
            # 设置箱线图颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 设置均值点样式
            bp['means'][0].set(marker='o', markerfacecolor='white', markeredgecolor='black')
        
        ax7.set_ylabel(LABELS['sentiment_intensity'], fontsize=10)
        ax7.set_title(LABELS['topic_boxplot'], fontsize=12, fontweight='bold')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 添加箱线图说明
        boxplot_explanation = "Boxplot Explanation:\n• Box: 25%-75% data range\n• Middle line: Median\n• White dot: Mean\n• Whiskers: Data range"
        
        ax7.text(0.02, 0.98, boxplot_explanation,
                transform=ax7.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 8. 主题置信度分布
        ax8 = plt.subplot(3, 3, 8)
        
        # 计算每个主题的平均置信度
        topic_confidence_stats = []
        
        for topic_id in range(num_topics):
            topic_mask = df['dominant_topic'] == topic_id
            if topic_mask.sum() > 0:
                conf_data = df.loc[topic_mask, 'topic_confidence'].values
                ax8.hist(conf_data, bins=20, alpha=0.5, color=colors[topic_id],
                        label=f'Topic {topic_id+1}', density=True)
                
                # 计算平均置信度
                avg_conf = conf_data.mean()
                topic_confidence_stats.append((topic_id, avg_conf))
        
        ax8.set_xlabel(LABELS['topic_confidence'], fontsize=10)
        ax8.set_ylabel('Density', fontsize=10)
        ax8.set_title(LABELS['confidence_distribution'], fontsize=12, fontweight='bold')
        
        if num_topics <= 6:
            ax8.legend(loc='upper right', fontsize=8)
        
        ax8.grid(True, alpha=0.3)
        
        # 添加置信度统计
        if topic_confidence_stats:
            conf_text = "Average Confidence:\n"
            for topic_id, avg_conf in topic_confidence_stats:
                conf_text += f"• Topic {topic_id+1}: {avg_conf:.3f}\n"
            
            ax8.text(0.02, 0.98, conf_text,
                    transform=ax8.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 9. 情感分类饼图
        ax9 = plt.subplot(3, 3, 9)
        sentiment_counts = df['sentiment'].value_counts()
        
        # 确保顺序一致
        sentiments = ['positive', 'negative', 'neutral']
        counts = [sentiment_counts.get(s, 0) for s in sentiments]
        
        if USE_ENGLISH_LABELS:
            labels = ['Positive', 'Negative', 'Neutral']
        else:
            labels = [LABELS['positive'], LABELS['negative'], LABELS['neutral']]
        
        colors_pie = ['lightgreen', 'lightcoral', 'lightblue']
        
        wedges, texts, autotexts = ax9.pie(counts, 
                                          labels=labels,
                                          autopct='%1.1f%%',
                                          colors=colors_pie,
                                          startangle=90,
                                          explode=(0.05, 0.05, 0.05))
        
        # 美化百分比文本
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax9.set_title(LABELS['sentiment_pie'], fontsize=12, fontweight='bold')
        
        # 添加具体数量注释
        total_lines = len(df)
        pie_info = f"Counts:\n• Positive: {counts[0]}\n• Negative: {counts[1]}\n• Neutral: {counts[2]}\n\nTotal: {total_lines}"
        
        ax9.text(-1.5, -1.2, pie_info, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle(LABELS['dashboard_title'], fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图形
        if USE_ENGLISH_LABELS:
            plt.savefig('anime_dialogue_analysis_english.png', dpi=150, bbox_inches='tight')
            print("英文可视化图表已保存为 'anime_dialogue_analysis_english.png'")
        else:
            plt.savefig('anime_dialogue_analysis_chinese.png', dpi=150, bbox_inches='tight')
            print("中文可视化图表已保存为 'anime_dialogue_analysis_chinese.png'")
        
        plt.show()
        
        # 如果有角色数据，创建额外图表
        character_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['character', 'role', 'speaker'])]
        
        if character_cols:
            character_col = character_cols[0]
            if df[character_col].dtype == 'object' and df[character_col].nunique() < 20:
                plt.figure(figsize=(14, 8))
                character_sentiment = df.groupby(character_col)['compound'].agg(['mean', 'count', 'std']).round(3)
                character_sentiment = character_sentiment.sort_values('mean', ascending=False)
                
                # 角色情感条形图
                colors_char = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(character_sentiment)))
                bars = plt.barh(range(len(character_sentiment)), character_sentiment['mean'].values, 
                               color=colors_char, edgecolor='black', alpha=0.8)
                
                plt.xlabel('Average Sentiment Intensity (-1 to 1)', fontsize=11)
                plt.ylabel('Character', fontsize=11)
                plt.yticks(range(len(character_sentiment)), character_sentiment.index, fontsize=10)
                plt.title('Character Sentiment Analysis', fontsize=14, fontweight='bold')
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
                
                # 添加数值标签
                for i, (bar, row) in enumerate(zip(bars, character_sentiment.iterrows())):
                    character_name, data = row
                    val = data['mean']
                    count = data['count']
                    color = 'black' if abs(val) < 0.5 else 'white'
                    
                    # 显示平均情感和台词数量
                    label_text = f'{val:.3f} ({count} lines)'
                    plt.text(val + (0.01 if val >= 0 else -0.08), bar.get_y() + bar.get_height()/2,
                            label_text, color=color, va='center',
                            fontweight='bold' if abs(val) > 0.3 else 'normal', fontsize=9)
                
                plt.grid(True, alpha=0.3, axis='x')
                plt.legend(fontsize=10)
                
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
                example_line = "N/A"
        else:
            example_line = "N/A"
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'size_pct': cluster_size / len(df),
            'avg_sentiment': avg_sentiment,
            'dominant_topic': dominant_topic,
            'topic_pct': topic_pct,
            'example': example_line
        })
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_size} ({cluster_size/len(df):.1%})")
        print(f"  Avg sentiment: {avg_sentiment:.3f}")
        if dominant_topic != -1:
            print(f"  Main topic: Topic {dominant_topic + 1} ({topic_pct:.1%})")
        print(f"  Representative dialogue: {example_line}")
    
    # 可视化聚类
    if len(df) > 30 and best_k > 1:
        try:
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 4))
            X_tsne = tsne.fit_transform(X_cluster)
            
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=df['cluster'], cmap='tab10',
                                 alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
            
            plt.colorbar(scatter, label='Cluster ID')
            plt.xlabel('t-SNE Dimension 1', fontsize=11)
            plt.ylabel('t-SNE Dimension 2', fontsize=11)
            plt.title(f'Dialogue Clustering Visualization (k={best_k})', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
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
    
    # 生成报告文本（使用英文，避免乱码）
    report = f"""
{'='*80}
Anime Dialogue Text Analysis Report
{'='*80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Dataset Overview
{'─'*40}
• Total dialogues: {len(df):,}
• Total words: {vocab_stats['total_words']:,}
• Unique words: {vocab_stats['unique_words']:,}
• Lexical diversity: {vocab_stats['lexical_diversity']:.2%}

2. Dialogue Length Analysis
{'─'*40}
• Average character length: {length_stats['avg_chars']:.1f} (±{length_stats['std_chars']:.1f})
• Average word count: {length_stats['avg_words']:.1f} (±{length_stats['std_words']:.1f})
• Longest dialogue: {length_stats['max_chars']} characters ({length_stats['max_words']} words)
• Shortest dialogue: {length_stats['min_chars']} characters ({length_stats['min_words']} words)

3. Sentiment Analysis Results
{'─'*40}
• Positive dialogues: {sentiment_stats['positive']:,} ({sentiment_stats['positive_pct']:.1%})
• Negative dialogues: {sentiment_stats['negative']:,} ({sentiment_stats['negative_pct']:.1%})
• Neutral dialogues: {sentiment_stats['neutral']:,} ({sentiment_stats['neutral_pct']:.1%})
• Average sentiment intensity: {sentiment_stats['avg_compound_abs']:.3f}
• Overall sentiment tone: {"Positive" if sentiment_stats['avg_compound'] > 0.1 else "Negative" if sentiment_stats['avg_compound'] < -0.1 else "Neutral"}
• Sentiment turning points: {len(turning_points)} 

4. Topic Modeling Findings
{'─'*40}
Found {len(topics_dict)} topics:

"""
    
    # 添加主题详情
    for topic_name, words in topics_dict.items():
        topic_id = int(topic_name.split('_')[1]) - 1
        topic_count = (df['dominant_topic'] == topic_id).sum()
        topic_pct = topic_count / len(df)
        
        # 计算该主题的平均情感
        topic_sentiment = df[df['dominant_topic'] == topic_id]['compound'].mean()
        sentiment_label = "Positive" if topic_sentiment > 0.1 else "Negative" if topic_sentiment < -0.1 else "Neutral"
        
        report += f"• {topic_name}: {', '.join(words[:8])}\n"
        report += f"  Count: {topic_count} dialogues ({topic_pct:.1%}) | "
        report += f"Avg sentiment: {topic_sentiment:.3f} ({sentiment_label})\n\n"
    
    # 添加聚类分析结果
    if cluster_summary:
        report += f"5. Clustering Analysis Results\n{'─'*40}\n"
        report += f"Found {len(cluster_summary)} clusters:\n\n"
        
        for cluster in cluster_summary:
            report += f"• Cluster {cluster['cluster_id']}:\n"
            report += f"  Size: {cluster['size']} dialogues ({cluster['size_pct']:.1%})\n"
            report += f"  Avg sentiment: {cluster['avg_sentiment']:.3f}\n"
            if cluster['dominant_topic'] != -1:
                report += f"  Main topic: Topic {cluster['dominant_topic'] + 1} ({cluster['topic_pct']:.1%})\n"
            report += f"  Representative dialogue: {cluster['example']}\n\n"
    
    # 添加高频词汇
    report += f"6. High Frequency Words Analysis\n{'─'*40}\n"
    report += "Top 15 content words (stopwords removed):\n\n"
    
    if vocab_stats['top_content_words']:
        for i, (word, freq) in enumerate(list(vocab_stats['top_content_words'].items())[:15], 1):
            freq_pct = freq / vocab_stats['total_words'] if vocab_stats['total_words'] > 0 else 0
            report += f"{i:2d}. {word:15s}: {freq:5d} times ({freq_pct:.2%})\n"
    else:
        report += "No sufficient high-frequency word data\n"
    
    # 添加叙事洞察
    report += f"\n7. Narrative Insights\n{'─'*40}\n"
    
    # 分析情感变化模式
    if len(turning_points) > 0:
        report += "• The emotional narrative has obvious fluctuations, with {} turning points found\n".format(len(turning_points))
        
        # 分析前半段和后半段的情感
        if len(df) > 10:
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]['compound'].mean()
            second_half = df.iloc[mid_point:]['compound'].mean()
            
            if abs(second_half - first_half) > 0.2:
                if second_half > first_half:
                    report += "• The second half of the narrative is more positive than the first half, showing an upward trend\n"
                else:
                    report += "• The second half of the narrative is more negative than the first half, showing a downward trend\n"
    
    # 添加分析建议
    report += f"\n8. Analysis Suggestions\n{'─'*40}\n"
    report += "1. Character identification: Use named entity recognition to automatically identify speakers\n"
    report += "2. Context analysis: Consider the emotional transmission relationship between dialogues\n"
    report += "3. Cultural context: Add more Japanese anime-specific vocabulary to the sentiment lexicon\n"
    report += "4. Multi-work comparison: Analyze dialogue style differences between different directors or periods\n"
    report += "5. Narrative structure: Combine with classical narrative theory to analyze structure\n"
    
    # 添加技术说明
    report += f"\n9. Technical Specifications\n{'─'*40}\n"
    report += "• Sentiment analysis: Using NLTK's VADER, extended with anime-related vocabulary\n"
    report += "• Topic modeling: Using LDA algorithm, TF-IDF vectorization\n"
    report += "• Clustering analysis: Using K-Means algorithm, silhouette coefficient selects optimal K value\n"
    report += "• Visualization: Using Matplotlib and Seaborn to create comprehensive dashboard\n"
    report += "• File support: Supports .txt, .csv, .xlsx, .xls formats\n"
    
    report += f"\n{'='*80}\nAnalysis Complete!\n{'='*80}"
    
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
    
    # 您的文件路径
    file_path = (r"E:\zhishitupukejian\作业\龙猫1988\副本龙猫.xlsx",
                 r"E:\zhishitupukejian\作业\哈尔的移动城堡2004\副本哈尔的移动城堡.xlsx",
                 r"E:\zhishitupukejian\作业\你想活出怎样的人生2023\副本你想活出怎样的人生.xlsx",
                 r"E:\zhishitupukejian\作业\崖上的金姬鱼2008\副本崖上的金姬鱼.xlsx",
                 r"E:\zhishitupukejian\作业\红猪1992\副本红猪.xlsx",
                 r"E:\zhishitupukejian\作业\借东西的小人阿莉埃蒂2010\副本借东西的小人阿莉埃蒂.xlsx",
                 r"E:\zhishitupukejian\作业\魔女宅急便1989\副本魔女宅急便.xlsx",
                 r"E:\zhishitupukejian\作业\起风了2013\副本起风了.xlsx",
                 r"E:\zhishitupukejian\作业\千与千寻2001\副本千与千寻.xlsx",
                 r"E:\zhishitupukejian\作业\天空之城1986\副本天空之城.xlsx",
                 r"E:\zhishitupukejian\作业\幽灵公主1997",
                 r"E:\微信数据\xwechat_files\wxid_o2khe2x3f3w522_3df9\msg\file\2025-12\风之谷.xlsx")  # 您的Excel文件路径
    
    # Excel文件中台词文本所在的列名
    text_column = "dialogue"  # 如果您的Excel中台词列不是'dialogue'，请修改
    
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
        print("="*60)
        print("生成的文件:")
        
        if USE_ENGLISH_LABELS:
            print("  1. anime_dialogue_analysis_english.png - 英文可视化图表")
        else:
            print("  1. anime_dialogue_analysis_chinese.png - 中文可视化图表")
        
        print("  2. anime_dialogue_analysis_report.txt - 详细分析报告")
        print("  3. anime_dialogue_analysis_results.csv - 分析结果数据")
        print("  4. anime_dialogue_topics.csv - 主题详情")
        
        # 根据条件显示其他文件
        character_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['character', 'role', 'speaker'])]
        if character_cols and df[character_cols[0]].nunique() < 20:
            print("  5. character_sentiment_analysis.png - 角色情感分析")
        
        if len(df) > 30 and cluster_summary:
            print("  6. clustering_visualization.png - 聚类可视化")
        
        print("="*60)
        
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
    # 检查依赖库
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
    
    # 显示语言设置
    if USE_ENGLISH_LABELS:
        print("当前使用英文标签（避免乱码）")
        print("如需使用中文标签，请将代码第30行的 USE_ENGLISH_LABELS = True 改为 False")
    else:
        print("当前使用中文标签")
        print("如果出现乱码，请将代码第30行的 USE_ENGLISH_LABELS = False 改为 True")
    
    print("="*60)
    
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

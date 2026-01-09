# ============================================
# 动漫台词文本数据分析完整流程
# 版本：3.0 - 支持批量分析多个文件
# 功能：从多个Excel/TXT文件到综合深度分析报告的全流程
# ============================================

# 第一部分：导入所有必要的库
print("正在导入库...")

import os
import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import glob
from datetime import datetime
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

# 先下载NLTK数据
print("下载NLTK数据...")
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTK数据下载失败: {e}")
    print("尝试使用本地缓存...")

# 设置中文字体 - 修复版本
def setup_chinese_font():
    """设置中文字体显示，解决中文乱码问题"""
    import matplotlib
    from matplotlib import font_manager
    
    # 尝试不同的方法设置中文字体
    try:
        # 方法1：使用系统字体
        system_fonts = font_manager.findSystemFonts()
        
        # 查找中文字体
        chinese_fonts = []
        for font in system_fonts:
            try:
                font_name = font_manager.FontProperties(fname=font).get_name()
                # 检查是否是中文字体
                if any(keyword in font_name.lower() for keyword in ['hei', 'song', 'kai', 'fang', 'yahei', 'yuan', 'ming']):
                    chinese_fonts.append(font)
            except:
                continue
        
        print(f"找到 {len(chinese_fonts)} 个中文字体")
        
        if chinese_fonts:
            # 使用找到的第一个中文字体
            font_path = chinese_fonts[0]
            font_prop = font_manager.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            
            # 添加到字体管理器
            font_manager.fontManager.addfont(font_path)
            
            # 设置matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            print(f"成功设置中文字体: {font_name}")
            
            # 验证字体设置
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '中文测试', fontproperties=font_prop, ha='center', va='center')
            ax.axis('off')
            plt.close(fig)
            
            return True
        
    except Exception as e:
        print(f"字体设置遇到问题: {e}")
    
    # 方法2：备用方案 - 设置通用字体
    try:
        # 尝试不同的通用中文字体名称
        font_candidates = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                          'DejaVu Sans', 'sans-serif', 'SimSun', 'NSimSun', 
                          'FangSong', 'KaiTi', 'STXihei', 'STKaiti']
        
        plt.rcParams['font.sans-serif'] = font_candidates
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"使用备用字体方案: {font_candidates[:3]}")
        return True
        
    except Exception as e:
        print(f"备用字体方案失败: {e}")
    
    print("警告: 无法设置中文字体，图表中的中文可能显示为方框")
    return False

# 在代码执行前设置中文字体
font_setup_result = setup_chinese_font()

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("所有库导入完成！\n" + "="*60)

# ============================================
# 第二部分：数据加载与预处理（支持批量文件）
# ============================================

def load_single_file(file_path, text_column='dialogue', file_id=None):
    """
    加载单个文件的台词数据
    
    参数:
        file_path: 文件路径
        text_column: Excel文件中台词文本所在的列名
        file_id: 文件标识符
    """
    print(f"正在加载文件: {os.path.basename(file_path)}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件: {file_path}")
        return None
    
    # 根据文件扩展名选择加载方式
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    
    if file_ext in ['.xlsx', '.xls']:
        # Excel文件处理
        try:
            # 尝试读取Excel文件
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                # 尝试不同的引擎
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except:
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except:
                        print(f"无法读取Excel文件 {file_name}: {e}")
                        return None
            
            # 自动检测台词列
            if text_column not in df.columns:
                # 尝试自动检测台词列
                possible_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    # 检查是否包含常见的关键词
                    if any(keyword in col_lower for keyword in 
                          ['text', 'dialogue', 'line', '台词', '对话', '脚本', '台词内容', '内容', 'cueline']):
                        possible_cols.append(col)
                    elif df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                        # 如果列是文本类型，检查样本内容
                        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ''
                        if isinstance(sample, str) and len(sample) > 10:
                            possible_cols.append(col)
                
                if possible_cols:
                    text_column = possible_cols[0]
                    print(f"  检测到台词列: {text_column}")
                else:
                    # 选择第一个文本列
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if text_cols:
                        text_column = text_cols[0]
                        print(f"  使用第一个文本列: {text_column}")
                    else:
                        # 如果没有文本列，使用第一列
                        text_column = df.columns[0]
                        print(f"  使用第一列: {text_column}")
            
            # 提取台词文本
            lines = df[text_column].fillna('').astype(str).tolist()
            raw_df = df.copy()
            
        except Exception as e:
            print(f"读取Excel文件 {file_name} 失败: {e}")
            return None
            
    elif file_ext == '.txt':
        # 文本文件处理
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"  使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # 所有编码都失败
                print(f"无法读取文本文件 {file_name}: 编码问题")
                return None
                
            raw_df = None
            
        except Exception as e:
            print(f"读取文本文件 {file_name} 失败: {e}")
            return None
                
    elif file_ext == '.csv':
        # CSV文件处理
        try:
            # 尝试多种编码读取CSV
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"  使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"无法读取CSV文件 {file_name}: 编码问题")
                return None
            
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
                    print(f"  检测到台词列: {text_column}")
                else:
                    text_column = df.columns[0]
                    print(f"  使用第一列: {text_column}")
            
            lines = df[text_column].fillna('').astype(str).tolist()
            raw_df = df.copy()
            
        except Exception as e:
            print(f"读取CSV文件 {file_name} 失败: {e}")
            return None
            
    else:
        print(f"错误: 不支持的文件格式: {file_ext}")
        return None
    
    # 创建基础数据框
    df_base = pd.DataFrame({
        'raw_line': lines,
        'line_number': range(1, len(lines) + 1),
        'file_name': file_name,
        'file_path': file_path,
        'file_id': file_id if file_id else file_name
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
    
    print(f"  清洗后有效台词数: {len(df_base)} (移除了 {initial_count - len(df_base)} 行)")
    
    # 计算基本长度指标
    df_base['char_length'] = df_base['cleaned_line'].apply(len)
    df_base['word_count'] = df_base['cleaned_line'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
    
    # 如果是从结构化文件加载的，尝试合并元数据
    if raw_df is not None and len(df_base) > 0:
        try:
            # 确保长度匹配
            if len(df_base) == len(raw_df):
                # 排除已存在的列和台词列
                exclude_cols = ['raw_line', 'cleaned_line', 'char_length', 'word_count', 
                               'file_name', 'file_path', 'file_id', text_column]
                
                for col in raw_df.columns:
                    if col not in exclude_cols and col not in df_base.columns:
                        # 检查列类型，确保可以合并
                        if len(df_base) == len(raw_df[col]):
                            df_base[col] = raw_df[col].values
                        
                print(f"  合并了 {len([c for c in df_base.columns if c not in ['raw_line', 'line_number', 'cleaned_line', 'char_length', 'word_count', 'file_name', 'file_path', 'file_id']])} 个元数据列")
        except Exception as e:
            print(f"  合并元数据时出错: {e}")
    
    return df_base


def load_multiple_files(file_paths, text_column='dialogue'):
    """
    批量加载多个文件
    
    参数:
        file_paths: 文件路径列表
        text_column: Excel文件中台词文本所在的列名
    """
    print(f"批量加载 {len(file_paths)} 个文件...")
    all_data = []
    
    for i, file_path in enumerate(file_paths):
        df = load_single_file(file_path, text_column, file_id=f"file_{i+1}")
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"  文件 {i+1}/{len(file_paths)} 加载完成: {os.path.basename(file_path)} ({len(df)} 行)")
        else:
            print(f"  文件 {i+1}/{len(file_paths)} 加载失败或为空: {os.path.basename(file_path)}")
    
    if not all_data:
        print("错误: 没有成功加载任何文件")
        return None, None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 添加全局行号
    combined_df['global_line_number'] = range(1, len(combined_df) + 1)
    
    print(f"\n批量加载完成！")
    print(f"总文件数: {len(all_data)}")
    print(f"总台词数: {len(combined_df):,}")
    print(f"平均每文件台词数: {len(combined_df) / len(all_data):.0f}")
    
    # 统计每个文件的信息
    file_stats = combined_df.groupby('file_name').agg({
        'global_line_number': 'count',
        'word_count': ['mean', 'sum'],
        'char_length': ['mean', 'max']
    }).round(1)
    
    file_stats.columns = ['台词数', '平均词数', '总词数', '平均字符数', '最大字符数']
    print("\n各文件统计:")
    print(file_stats.to_string())
    
    return combined_df, file_stats


# ============================================
# 第三部分：词汇统计分析（支持批量）
# ============================================

def analyze_vocabulary(df, by_file=False):
    """
    分析词汇使用情况
    
    参数:
        df: 数据框
        by_file: 是否按文件分组分析
    """
    print("\n" + "="*60)
    print("开始词汇统计分析...")
    
    if by_file and 'file_name' in df.columns:
        # 按文件分析
        print("按文件分析词汇...")
        file_vocab_stats = {}
        
        for file_name in df['file_name'].unique():
            file_df = df[df['file_name'] == file_name]
            all_text = " ".join(file_df['cleaned_line'].tolist())
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            
            word_freq = Counter(words)
            total_words = len(words)
            unique_words = len(word_freq)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # 过滤停用词
            stop_words = set(stopwords.words('english'))
            content_words = {word: freq for word, freq in word_freq.items() 
                            if word not in stop_words}
            top_content_words = dict(sorted(content_words.items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
            
            file_vocab_stats[file_name] = {
                'total_words': total_words,
                'unique_words': unique_words,
                'lexical_diversity': lexical_diversity,
                'top_words': top_content_words
            }
        
        # 整体分析
        all_text = " ".join(df['cleaned_line'].tolist())
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        word_freq = Counter(words)
        total_words = len(words)
        unique_words = len(word_freq)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # 过滤停用词后的高频词
        stop_words = set(stopwords.words('english'))
        content_words = {word: freq for word, freq in word_freq.items() 
                        if word not in stop_words}
        top_content_words = dict(sorted(content_words.items(), 
                                       key=lambda x: x[1], reverse=True)[:20])
        
        stats = {
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity,
            'word_freq': word_freq,
            'top_content_words': top_content_words,
            'by_file': file_vocab_stats
        }
        
        # 输出文件比较
        print("\n各文件词汇多样性比较:")
        for file_name, file_stats in file_vocab_stats.items():
            short_name = file_name[:30] + "..." if len(file_name) > 30 else file_name
            print(f"  {short_name:35s}: 词汇数={file_stats['total_words']:5d}, "
                  f"独特性={file_stats['lexical_diversity']:.3%}")
        
    else:
        # 整体分析
        all_text = " ".join(df['cleaned_line'].tolist())
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
            'top_content_words': top_content_words,
            'by_file': None
        }
    
    print(f"\n整体统计:")
    print(f"总词数: {stats['total_words']:,}")
    print(f"独特词数: {stats['unique_words']:,}")
    print(f"词汇多样性: {stats['lexical_diversity']:.3%}")
    print(f"前10个高频词（排除停用词）:")
    for i, (word, freq) in enumerate(list(stats['top_content_words'].items())[:10], 1):
        print(f"  {i:2d}. {word:15s}: {freq:5d}次")
    
    return stats


# ============================================
# 第四部分：情感分析（支持批量）
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
    
    # 按文件计算移动平均（如果数据量大）
    if 'file_name' in df.columns and df['file_name'].nunique() > 1:
        df['compound_ma'] = 0
        for file_name in df['file_name'].unique():
            file_mask = df['file_name'] == file_name
            file_len = file_mask.sum()
            if file_len > 10:
                window_size = min(20, file_len // 10)
                if window_size % 2 == 0:
                    window_size += 1
                
                df.loc[file_mask, 'compound_ma'] = df.loc[file_mask, 'compound'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            else:
                df.loc[file_mask, 'compound_ma'] = df.loc[file_mask, 'compound']
    else:
        # 整体计算移动平均
        window_size = min(20, len(df) // 10)
        if window_size % 2 == 0:
            window_size += 1
        
        df['compound_ma'] = df['compound'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
    
    # 输出整体情感统计
    positive_pct = (df['sentiment'] == 'positive').sum() / len(df)
    negative_pct = (df['sentiment'] == 'negative').sum() / len(df)
    neutral_pct = (df['sentiment'] == 'neutral').sum() / len(df)
    
    print(f"整体情感统计:")
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
    
    # 按文件输出情感统计
    if 'file_name' in df.columns:
        print("\n各文件情感统计:")
        file_sentiment_stats = []
        
        for file_name in df['file_name'].unique():
            file_df = df[df['file_name'] == file_name]
            file_positive = (file_df['sentiment'] == 'positive').sum() / len(file_df)
            file_negative = (file_df['sentiment'] == 'negative').sum() / len(file_df)
            file_neutral = (file_df['sentiment'] == 'neutral').sum() / len(file_df)
            file_avg_sentiment = file_df['compound'].mean()
            
            file_sentiment_stats.append({
                'file_name': file_name,
                'positive': file_positive,
                'negative': file_negative,
                'neutral': file_neutral,
                'avg_sentiment': file_avg_sentiment,
                'line_count': len(file_df)
            })
            
            short_name = file_name[:25] + "..." if len(file_name) > 25 else file_name
            tone = "积极" if file_avg_sentiment > 0.1 else "消极" if file_avg_sentiment < -0.1 else "中性"
            print(f"  {short_name:28s}: 积极{file_positive:5.1%} 消极{file_negative:5.1%} "
                  f"中性{file_neutral:5.1%} ({tone})")
    
    # 识别情感转折点
    turning_points = []
    for i in range(1, len(df) - 1):
        if pd.notna(df.loc[i, 'compound_ma']) and pd.notna(df.loc[i-1, 'compound_ma']):
            prev = df.loc[i-1, 'compound_ma']
            curr = df.loc[i, 'compound_ma']
            next_val = df.loc[i+1, 'compound_ma'] if i+1 < len(df) else curr
            
            # 检测峰值
            if prev < curr > next_val and (curr - prev) > 0.3:
                file_name = df.loc[i, 'file_name'] if 'file_name' in df.columns else "整体"
                turning_points.append(('peak', i, curr, df.loc[i, 'cleaned_line'][:50], file_name))
            # 检测谷值
            elif prev > curr < next_val and (prev - curr) > 0.3:
                file_name = df.loc[i, 'file_name'] if 'file_name' in df.columns else "整体"
                turning_points.append(('valley', i, curr, df.loc[i, 'cleaned_line'][:50], file_name))
    
    print(f"发现情感转折点: {len(turning_points)} 个")
    
    return df, turning_points


# ============================================
# 第五部分：主题建模（支持批量）
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
    
    # 分析整体主题分布
    topic_counts = df['dominant_topic'].value_counts().sort_index()
    print("\n整体主题分布:")
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
    
    # 按文件分析主题分布
    if 'file_name' in df.columns:
        print("\n各文件主题分布:")
        file_topic_stats = {}
        
        for file_name in df['file_name'].unique():
            file_df = df[df['file_name'] == file_name]
            file_topic_counts = file_df['dominant_topic'].value_counts().sort_index()
            
            file_topic_stats[file_name] = {}
            for topic_id in range(num_topics):
                count = file_topic_counts.get(topic_id, 0)
                pct = count / len(file_df) if len(file_df) > 0 else 0
                file_topic_stats[file_name][f'topic_{topic_id+1}'] = pct
            
            short_name = file_name[:20] + "..." if len(file_name) > 20 else file_name
            topic_str = " ".join([f"T{t+1}:{file_topic_stats[file_name][f'topic_{t+1}']:.0%}" 
                                 for t in range(num_topics)])
            print(f"  {short_name:23s}: {topic_str}")
    
    return df, topics_dict, lda, X, vectorizer


# ============================================
# ============================================
# ============================================
# 第六部分：可视化分析（支持批量） - 优化版本
# 修复了注释文字乱码问题
# ============================================

def create_visualizations(df, topics_dict, turning_points, vocab_stats, file_stats=None):
    """
    创建所有可视化图表，修复字体乱码问题
    """
    print("\n" + "="*60)
    print("创建可视化图表...")
    
    try:
        # 设置中文字体 - 修复保存图片时的字体问题
        import matplotlib
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        
        # 重新设置字体，确保保存时使用正确字体
        def setup_font_for_saving():
            """为保存图片专门设置字体"""
            try:
                # 查找中文字体
                font_path = None
                
                # 常见中文字体路径
                common_fonts = [
                    # Windows
                    r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
                    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
                    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
                    # macOS
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",
                    # Linux
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                ]
                
                for font in common_fonts:
                    if os.path.exists(font):
                        font_path = font
                        break
                
                if font_path:
                    # 添加到字体管理器
                    font_manager.fontManager.addfont(font_path)
                    font_name = font_manager.FontProperties(fname=font_path).get_name()
                    
                    # 设置matplotlib参数
                    matplotlib.rcParams['font.sans-serif'] = [font_name]
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    
                    # 验证字体
                    test_fig, test_ax = plt.subplots(figsize=(1, 1))
                    test_ax.text(0.5, 0.5, '测试', fontproperties=font_manager.FontProperties(fname=font_path))
                    plt.close(test_fig)
                    
                    print(f"保存字体设置: {font_name}")
                    return font_path, font_name
                else:
                    # 使用系统已安装字体
                    font_names = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                                 'DejaVu Sans', 'sans-serif', 'SimSun']
                    matplotlib.rcParams['font.sans-serif'] = font_names
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    print(f"使用备用字体: {font_names[0]}")
                    return None, font_names[0]
                    
            except Exception as e:
                print(f"字体设置失败: {e}")
                return None, None
        
        # 为保存图片设置字体
        font_path, font_name = setup_font_for_saving()
        
        # 根据数据量调整图形大小
        num_files = df['file_name'].nunique() if 'file_name' in df.columns else 1
        
        if num_files > 6:
            fig = plt.figure(figsize=(22, 20))
        elif num_files > 3:
            fig = plt.figure(figsize=(20, 18))
        else:
            fig = plt.figure(figsize=(20, 16))
        
        # 设置全局字体大小
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        })
        
        # 如果需要，为每个文本对象设置字体属性
        if font_path:
            chinese_font_prop = font_manager.FontProperties(fname=font_path)
        else:
            chinese_font_prop = None
        
        # 1. 情感叙事曲线（按文件） - 优化布局
        ax1 = plt.subplot(3, 3, 1)
        
        if 'file_name' in df.columns and df['file_name'].nunique() > 1:
            colors = plt.cm.tab20c(np.linspace(0, 1, min(20, df['file_name'].nunique())))
            files_to_show = min(8, df['file_name'].nunique())
            files_displayed = 0
            file_sizes = df.groupby('file_name').size()
            largest_files = file_sizes.nlargest(files_to_show).index.tolist()
            
            for idx, file_name in enumerate(largest_files):
                file_df = df[df['file_name'] == file_name]
                if len(file_df) > 0:
                    color = colors[idx % len(colors)]
                    short_name = file_name[:12] + "..." if len(file_name) > 12 else file_name
                    
                    if 'compound_ma' in file_df.columns and file_df['compound_ma'].notna().sum() > 0:
                        ax1.plot(file_df.index, file_df['compound_ma'], 
                                linewidth=1.2, alpha=0.7, color=color, label=short_name)
                        files_displayed += 1
            
            if files_displayed < df['file_name'].nunique():
                other_files = [f for f in df['file_name'].unique() if f not in largest_files]
                other_df = df[df['file_name'].isin(other_files)]
                if len(other_df) > 0 and 'compound_ma' in other_df.columns:
                    ax1.plot(other_df.index, other_df['compound_ma'],
                            linewidth=0.8, alpha=0.3, color='gray', 
                            label=f'其他 {len(other_files)}个文件')
        else:
            ax1.plot(df.index, df['compound'], alpha=0.4, linewidth=0.6, 
                    label='原始', color='lightblue')
            
            if 'compound_ma' in df.columns and df['compound_ma'].notna().sum() > 0:
                ax1.plot(df.index, df['compound_ma'], linewidth=1.5, 
                        label='平滑', color='red')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.set_xlabel('台词序列')
        ax1.set_ylabel('情感强度')
        ax1.set_title('情感叙事曲线', fontproperties=chinese_font_prop)
        
        if ax1.get_legend_handles_labels()[0]:
            ncol = 2 if files_displayed > 4 else 1
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                      ncol=ncol, fontsize=8, framealpha=0.7,
                      prop=chinese_font_prop if chinese_font_prop else None)
        
        ax1.grid(True, alpha=0.2)
        
        # 2. 情感分布直方图
        ax2 = plt.subplot(3, 3, 2)
        bins_count = min(30, max(10, len(df) // 50))
        n, bins, patches = ax2.hist(df['compound'], bins=bins_count, 
                                   edgecolor='black', alpha=0.7, 
                                   color='skyblue', orientation='horizontal')
        
        mean_val = df['compound'].mean()
        ax2.axhline(y=mean_val, color='red', linestyle='--', 
                   alpha=0.7, linewidth=1.5, label=f'均值: {mean_val:.3f}')
        
        ax2.set_xlabel('频率')
        ax2.set_ylabel('情感强度')
        ax2.set_title('情感强度分布', fontproperties=chinese_font_prop)
        
        # 使用Unicode文本，避免中文字体问题
        ax2.text(0.95, 0.95, f'均值: {mean_val:.3f}\n标准差: {df["compound"].std():.3f}',
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.2)
        
        # 3. 各文件台词数量对比 - 优化条形图
        ax3 = plt.subplot(3, 3, 3)
        
        if file_stats is not None and len(file_stats) > 1:
            file_names = file_stats.index.tolist()
            line_counts = file_stats['台词数'].tolist()
            
            max_files_to_show = 8
            if len(file_names) > max_files_to_show:
                indices = np.argsort(line_counts)[-max_files_to_show+1:]
                file_names = [file_names[i] for i in indices]
                line_counts = [line_counts[i] for i in indices]
                other_indices = [i for i in range(len(file_names)) if i not in indices]
                other_total = sum([line_counts[i] for i in other_indices])
                file_names.append('其他')
                line_counts.append(other_total)
            
            short_names = []
            for name in file_names:
                if len(name) > 15:
                    parts = re.split(r'[\\/._-]', name)
                    if len(parts) > 1:
                        short_name = parts[-1]
                        if len(short_name) > 12:
                            short_name = short_name[:12] + "..."
                    else:
                        short_name = name[:12] + "..."
                else:
                    short_name = name
                short_names.append(short_name)
            
            y_pos = np.arange(len(short_names))
            colors_bar = plt.cm.Set3(np.linspace(0, 1, len(short_names)))
            
            bars = ax3.barh(y_pos, line_counts, color=colors_bar, 
                           edgecolor='black', linewidth=0.5, height=0.7)
            
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(short_names, fontsize=9)
            ax3.set_xlabel('台词数量')
            ax3.set_title('各文件台词数量', fontproperties=chinese_font_prop)
            ax3.invert_yaxis()
            
            max_count = max(line_counts)
            for i, (bar, count) in enumerate(zip(bars, line_counts)):
                width = bar.get_width()
                if width > max_count * 0.3:
                    text_x = width * 0.95
                    text_color = 'white'
                    ha = 'right'
                else:
                    text_x = width + max_count * 0.01
                    text_color = 'black'
                    ha = 'left'
                
                ax3.text(text_x, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', va='center', ha=ha,
                        color=text_color, fontsize=9, fontweight='bold')
            
            ax3.grid(True, alpha=0.2, axis='x')
        else:
            ax3.hist(df['word_count'], bins=30, edgecolor='black', 
                    alpha=0.7, color='lightgreen', density=True)
            
            mean_val = df['word_count'].mean()
            median_val = df['word_count'].median()
            
            ax3.axvline(mean_val, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=f'均值: {mean_val:.1f}')
            ax3.axvline(median_val, color='blue', linestyle=':', 
                       linewidth=1.5, alpha=0.7, label=f'中位数: {median_val:.1f}')
            
            ax3.set_xlabel('台词词数')
            ax3.set_ylabel('密度')
            ax3.set_title('台词长度分布', fontproperties=chinese_font_prop)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.2)
        
        # 4. 主题分布饼图 - 简化标签
        ax4 = plt.subplot(3, 3, 4)
        
        if 'dominant_topic' in df.columns and df['dominant_topic'].nunique() > 1:
            topic_counts = df['dominant_topic'].value_counts().sort_index()
            
            if len(topic_counts) > 8:
                main_topics = topic_counts.head(7)
                other_count = topic_counts[7:].sum()
                main_topics = pd.concat([main_topics, pd.Series([other_count], index=['其他'])])
                labels = [f'T{i+1}' for i in main_topics.index[:-1]] + ['其他']
                sizes = main_topics.values
            else:
                labels = [f'T{i+1}' for i in topic_counts.index]
                sizes = topic_counts.values
            
            if len(sizes) > 1:
                explode = [0.05 if i == sizes.argmax() else 0 for i in range(len(sizes))]
            else:
                explode = [0]
            
            colors_topic = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            
            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return f'{val}\n({pct:.1f}%)'
                return my_autopct
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, 
                                              autopct=make_autopct(sizes),
                                              colors=colors_topic, 
                                              startangle=90, explode=explode,
                                              textprops={'fontsize': 9})
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
            
            ax4.set_title('主题分布', fontproperties=chinese_font_prop)
        else:
            ax4.text(0.5, 0.5, '无主题数据', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
            ax4.set_title('主题分布', fontproperties=chinese_font_prop)
        
        # 5. 高频词词云风格 - 改为水平条形图
        ax5 = plt.subplot(3, 3, 5)
        
        if vocab_stats['top_content_words']:
            top_words = list(vocab_stats['top_content_words'].items())[:15]
            words, freqs = zip(*top_words)
            
            y_pos = np.arange(len(words))
            norm_freqs = np.array(freqs) / max(freqs)
            colors_word = plt.cm.Reds(0.3 + 0.7 * norm_freqs)
            
            bars = ax5.barh(y_pos, freqs, color=colors_word, 
                           edgecolor='black', linewidth=0.5, height=0.7)
            
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(words, fontsize=9)
            ax5.set_xlabel('出现频率')
            ax5.set_title('高频内容词 Top 15', fontproperties=chinese_font_prop)
            ax5.invert_yaxis()
            
            max_freq = max(freqs)
            for i, (bar, freq) in enumerate(zip(bars, freqs)):
                width = bar.get_width()
                if width > max_freq * 0.3:
                    text_x = width * 0.95
                    text_color = 'white'
                    ha = 'right'
                else:
                    text_x = width + max_freq * 0.01
                    text_color = 'black'
                    ha = 'left'
                
                rel_freq = freq / vocab_stats['total_words'] if vocab_stats['total_words'] > 0 else 0
                ax5.text(text_x, bar.get_y() + bar.get_height()/2,
                        f'{freq}\n({rel_freq:.2%})', va='center', ha=ha,
                        color=text_color, fontsize=8)
            
            ax5.grid(True, alpha=0.2, axis='x')
        else:
            ax5.text(0.5, 0.5, '无高频词数据', ha='center', va='center', 
                    fontsize=12, transform=ax5.transAxes)
            ax5.set_title('高频内容词', fontproperties=chinese_font_prop)
        
        # 6. 情感分类分布 - 使用英文标签避免乱码
        ax6 = plt.subplot(3, 3, 6)
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            
            # 使用英文标签，避免中文字体问题
            labels = []
            for label in sentiment_counts.index:
                if label == 'positive':
                    labels.append('Positive')
                elif label == 'negative':
                    labels.append('Negative')
                else:
                    labels.append('Neutral')
            
            sizes = sentiment_counts.values
            
            if len(sizes) == 3:
                explode = (0.05, 0.05, 0.05)
            elif len(sizes) == 2:
                explode = (0.05, 0.05)
            else:
                explode = [0.05] * len(sizes)
            
            colors_sentiment = ['#66c2a5', '#fc8d62', '#8da0cb'][:len(sizes)]
            
            wedges, texts, autotexts = ax6.pie(sizes, labels=labels, 
                                              autopct='%1.1f%%',
                                              colors=colors_sentiment,
                                              startangle=90, explode=explode,
                                              textprops={'fontsize': 9})
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
            
            ax6.set_title('情感分类比例', fontproperties=chinese_font_prop)
        else:
            ax6.text(0.5, 0.5, '无情感数据', ha='center', va='center', 
                    fontsize=12, transform=ax6.transAxes)
            ax6.set_title('情感分类比例', fontproperties=chinese_font_prop)
        
        # 7. 各文件平均情感对比
        ax7 = plt.subplot(3, 3, 7)
        
        if 'file_name' in df.columns and df['file_name'].nunique() > 1:
            file_sentiment = df.groupby('file_name')['compound'].agg(['mean', 'count']).sort_values('mean')
            max_files = 10
            if len(file_sentiment) > max_files:
                file_sentiment = file_sentiment.head(max_files)
            
            file_names = file_sentiment.index.tolist()
            sentiment_means = file_sentiment['mean'].tolist()
            file_counts = file_sentiment['count'].tolist()
            
            short_names = []
            for name in file_names:
                if len(name) > 12:
                    parts = re.split(r'[\\/._-]', name)
                    if len(parts) > 1:
                        short_name = parts[-1]
                        if len(short_name) > 10:
                            short_name = short_name[:10] + "..."
                    else:
                        short_name = name[:10] + "..."
                else:
                    short_name = name
                short_names.append(short_name)
            
            y_pos = np.arange(len(short_names))
            colors_sentiment_bar = []
            for val in sentiment_means:
                if val > 0.1:
                    colors_sentiment_bar.append('#66c2a5')
                elif val < -0.1:
                    colors_sentiment_bar.append('#fc8d62')
                else:
                    colors_sentiment_bar.append('#8da0cb')
            
            bars = ax7.barh(y_pos, sentiment_means, color=colors_sentiment_bar,
                           edgecolor='black', linewidth=0.5, height=0.7)
            
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(short_names, fontsize=9)
            ax7.set_xlabel('平均情感强度')
            ax7.set_title('各文件情感基调', fontproperties=chinese_font_prop)
            ax7.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            ax7.invert_yaxis()
            
            max_val = max(abs(min(sentiment_means)), abs(max(sentiment_means)))
            for i, (bar, val, count) in enumerate(zip(bars, sentiment_means, file_counts)):
                width = bar.get_width()
                if width > 0:
                    text_x = width + max_val * 0.02
                    ha = 'left'
                else:
                    text_x = width - max_val * 0.02
                    ha = 'right'
                
                ax7.text(text_x, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}\n({count}句)', va='center', ha=ha,
                        color='black', fontsize=8)
            
            ax7.grid(True, alpha=0.2, axis='x')
        else:
            if 'dominant_topic' in df.columns and df['dominant_topic'].nunique() > 1:
                num_topics = df['dominant_topic'].nunique()
                topic_data = []
                topic_labels = []
                
                for topic_id in range(num_topics):
                    topic_data.append(df[df['dominant_topic'] == topic_id]['compound'].values)
                    topic_labels.append(f'T{topic_id+1}')
                
                if all(len(data) > 0 for data in topic_data):
                    bp = ax7.boxplot(topic_data, labels=topic_labels, patch_artist=True)
                    colors_box = plt.cm.Set3(np.linspace(0, 1, num_topics))
                    for patch, color in zip(bp['boxes'], colors_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    for flier in bp['fliers']:
                        flier.set(marker='o', color='red', alpha=0.5, markersize=4)
                
                ax7.set_ylabel('情感强度')
                ax7.set_title('各主题情感分布', fontproperties=chinese_font_prop)
                ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
                ax7.grid(True, alpha=0.2, axis='y')
            else:
                ax7.text(0.5, 0.5, '无数据', ha='center', va='center', 
                        fontsize=12, transform=ax7.transAxes)
                ax7.set_title('主题情感分布', fontproperties=chinese_font_prop)
        
        # 8. 各文件词汇多样性对比
        ax8 = plt.subplot(3, 3, 8)
        
        if vocab_stats['by_file'] is not None and len(vocab_stats['by_file']) > 1:
            file_names = list(vocab_stats['by_file'].keys())
            lexical_diversities = [vocab_stats['by_file'][name]['lexical_diversity'] 
                                 for name in file_names]
            total_words = [vocab_stats['by_file'][name]['total_words'] 
                          for name in file_names]
            
            max_files = 8
            if len(file_names) > max_files:
                indices = np.argsort(lexical_diversities)[-max_files:]
                file_names = [file_names[i] for i in indices]
                lexical_diversities = [lexical_diversities[i] for i in indices]
                total_words = [total_words[i] for i in indices]
            
            short_names = []
            for name in file_names:
                if len(name) > 12:
                    parts = re.split(r'[\\/._-]', name)
                    if len(parts) > 1:
                        short_name = parts[-1]
                        if len(short_name) > 10:
                            short_name = short_name[:10] + "..."
                    else:
                        short_name = name[:10] + "..."
                else:
                    short_name = name
                short_names.append(short_name)
            
            y_pos = np.arange(len(short_names))
            norm_div = np.array(lexical_diversities) / max(lexical_diversities)
            colors_div = plt.cm.Blues(0.3 + 0.7 * norm_div)
            
            bars = ax8.barh(y_pos, lexical_diversities, color=colors_div,
                           edgecolor='black', linewidth=0.5, height=0.7)
            
            ax8.set_yticks(y_pos)
            ax8.set_yticklabels(short_names, fontsize=9)
            ax8.set_xlabel('词汇多样性指数')
            ax8.set_title('各文件词汇多样性', fontproperties=chinese_font_prop)
            ax8.invert_yaxis()
            
            max_div = max(lexical_diversities)
            for i, (bar, div, words) in enumerate(zip(bars, lexical_diversities, total_words)):
                width = bar.get_width()
                if width > max_div * 0.3:
                    text_x = width * 0.95
                    text_color = 'white'
                    ha = 'right'
                else:
                    text_x = width + max_div * 0.01
                    text_color = 'black'
                    ha = 'left'
                
                ax8.text(text_x, bar.get_y() + bar.get_height()/2,
                        f'{div:.3f}\n({words:,}词)', va='center', ha=ha,
                        color=text_color, fontsize=8)
            
            ax8.grid(True, alpha=0.2, axis='x')
        else:
            if 'word_count' in df.columns:
                word_counts = df['word_count']
                q99 = word_counts.quantile(0.99)
                word_counts_filtered = word_counts[word_counts <= q99]
                
                ax8.hist(word_counts_filtered, bins=20, edgecolor='black', 
                        alpha=0.7, color='lightgreen', density=True)
                
                mean_val = word_counts_filtered.mean()
                median_val = word_counts_filtered.median()
                
                ax8.axvline(mean_val, color='red', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'均值: {mean_val:.1f}')
                ax8.axvline(median_val, color='blue', linestyle=':', 
                           linewidth=1.5, alpha=0.7, label=f'中位数: {median_val:.1f}')
                
                ax8.set_xlabel('台词词数')
                ax8.set_ylabel('密度')
                ax8.set_title('词长分布（过滤异常值）', fontproperties=chinese_font_prop)
                ax8.legend(fontsize=8)
                ax8.grid(True, alpha=0.2)
            else:
                ax8.text(0.5, 0.5, '无数据', ha='center', va='center', 
                        fontsize=12, transform=ax8.transAxes)
                ax8.set_title('词汇多样性', fontproperties=chinese_font_prop)
        
        # 9. 情感强度与词长关系
        ax9 = plt.subplot(3, 3, 9)
        
        if 'word_count' in df.columns and 'compound' in df.columns:
            if len(df) > 500:
                sample_indices = np.random.choice(len(df), 500, replace=False)
                word_counts = df.iloc[sample_indices]['word_count'].values
                compounds = df.iloc[sample_indices]['compound'].values
            else:
                word_counts = df['word_count'].values
                compounds = df['compound'].values
            
            colors_scatter = []
            for val in compounds:
                if val > 0.1:
                    colors_scatter.append('#66c2a5')
                elif val < -0.1:
                    colors_scatter.append('#fc8d62')
                else:
                    colors_scatter.append('#8da0cb')
            
            sizes = 20 + 50 * np.abs(compounds)
            scatter = ax9.scatter(word_counts, compounds, 
                                 c=colors_scatter, s=sizes, alpha=0.6,
                                 edgecolors='black', linewidth=0.5)
            
            if len(word_counts) > 2:
                try:
                    z = np.polyfit(word_counts, compounds, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(word_counts), max(word_counts), 100)
                    y_line = p(x_line)
                    ax9.plot(x_line, y_line, color='red', linestyle='--', 
                            linewidth=1.5, alpha=0.7, label='趋势线')
                    
                    correlation = np.corrcoef(word_counts, compounds)[0, 1]
                    ax9.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                            transform=ax9.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass
            
            ax9.set_xlabel('台词词数')
            ax9.set_ylabel('情感强度')
            ax9.set_title('情感强度 vs. 台词长度', fontproperties=chinese_font_prop)
            ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            ax9.legend(fontsize=8)
            ax9.grid(True, alpha=0.2)
        else:
            ax9.text(0.5, 0.5, '无数据', ha='center', va='center', 
                    fontsize=12, transform=ax9.transAxes)
            ax9.set_title('情感 vs. 长度', fontproperties=chinese_font_prop)
        
        # 设置主标题（使用Unicode确保显示）
        plt.suptitle('Anime Dialogue Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=1.5, w_pad=1.5)
        
        # 保存图片时指定字体
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'anime_dialogue_analysis_{timestamp}.png'
        
        # 保存图片时使用额外的字体设置
        try:
            if font_path:
                # 使用指定字体保存
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
            else:
                # 尝试使用默认设置
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"保存图片时出错: {e}")
            # 尝试不使用字体属性
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        
        print(f"可视化图表已保存为 '{output_file}'")
        plt.show()
        
        # 如果有角色数据，创建角色情感分析图表
        character_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['character', 'role', 'speaker'])]
        
        if character_cols:
            character_col = character_cols[0]
            if df[character_col].dtype == 'object' and df[character_col].nunique() > 1:
                char_counts = df[character_col].value_counts()
                main_chars = char_counts.head(10).index.tolist()
                
                if len(main_chars) > 1:
                    plt.figure(figsize=(12, 8))
                    
                    char_data = []
                    char_labels = []
                    
                    for char in main_chars:
                        char_mask = df[character_col] == char
                        if char_mask.sum() > 0:
                            char_data.append(df[char_mask]['compound'].values)
                            char_labels.append(f'{char}\n({char_mask.sum()}句)')
                    
                    bp = plt.boxplot(char_data, labels=char_labels, patch_artist=True)
                    colors_char = plt.cm.tab20c(np.linspace(0, 1, len(char_data)))
                    for patch, color in zip(bp['boxes'], colors_char):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    for flier in bp['fliers']:
                        flier.set(marker='o', color='red', alpha=0.5, markersize=4)
                    
                    plt.xlabel('角色')
                    plt.ylabel('情感强度')
                    plt.title('主要角色情感分布')
                    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    plt.grid(True, alpha=0.2, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    character_file = f'character_sentiment_analysis_{timestamp}.png'
                    plt.savefig(character_file, dpi=150, bbox_inches='tight')
                    plt.show()
                    print(f"角色情感分析图表已保存为 '{character_file}'")
        
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
        
        # 按文件分析聚类分布
        file_distribution = {}
        if 'file_name' in df.columns:
            file_counts = df.loc[cluster_mask, 'file_name'].value_counts()
            for file_name, count in file_counts.items():
                file_pct = count / cluster_size
                file_distribution[file_name] = file_pct
        
        # 找到代表性台词（情感最强烈的）
        if cluster_size > 0:
            try:
                representative = df[cluster_mask].loc[df[cluster_mask]['compound'].abs().idxmax()]
                example_line = representative['cleaned_line']
                example_file = representative['file_name'] if 'file_name' in representative else "未知"
                if len(example_line) > 80:
                    example_line = example_line[:80] + "..."
            except:
                example_line = "无"
                example_file = "未知"
        else:
            example_line = "无"
            example_file = "未知"
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'size_pct': cluster_size / len(df),
            'avg_sentiment': avg_sentiment,
            'dominant_topic': dominant_topic,
            'topic_pct': topic_pct,
            'file_distribution': file_distribution,
            'example': example_line,
            'example_file': example_file
        })
        
        print(f"\n聚类 {cluster_id}:")
        print(f"  大小: {cluster_size} ({cluster_size/len(df):.1%})")
        print(f"  平均情感: {avg_sentiment:.3f}")
        if dominant_topic != -1:
            print(f"  主要主题: 主题 {dominant_topic + 1} ({topic_pct:.1%})")
        
        # 显示文件分布（如果有多个文件）
        if file_distribution and len(file_distribution) > 1:
            print(f"  文件分布:")
            for file_name, pct in list(file_distribution.items())[:3]:
                short_name = file_name[:20] + "..." if len(file_name) > 20 else file_name
                print(f"    {short_name}: {pct:.1%}")
        
        print(f"  代表性台词 ({example_file}): {example_line}")
    
    # 可视化聚类
    if len(df) > 30 and best_k > 1:
        try:
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 4))
            X_tsne = tsne.fit_transform(X_cluster)
            
            plt.figure(figsize=(10, 8))
            
            if 'file_name' in df.columns and df['file_name'].nunique() > 1:
                # 按文件着色
                unique_files = df['file_name'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(unique_files))))
                
                for idx, file_name in enumerate(unique_files):
                    file_mask = df['file_name'] == file_name
                    if file_mask.sum() > 0:
                        color = colors[idx % len(colors)]
                        short_name = file_name[:10] + "..." if len(file_name) > 10 else file_name
                        plt.scatter(X_tsne[file_mask, 0], X_tsne[file_mask, 1], 
                                   alpha=0.6, s=30, color=color, label=short_name)
                
                plt.legend(loc='upper right', fontsize=9, ncol=2, prop={'family': 'SimHei'})
            else:
                # 按聚类着色
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                     c=df['cluster'], cmap='tab10',
                                     alpha=0.6, s=30)
                plt.colorbar(scatter, label='聚类')
            
            plt.xlabel('t-SNE 维度 1', fontproperties='SimHei')
            plt.ylabel('t-SNE 维度 2', fontproperties='SimHei')
            plt.title(f'台词聚类可视化 (k={best_k})', fontproperties='SimHei')
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clustering_file = f'clustering_visualization_{timestamp}.png'
            plt.savefig(clustering_file, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"聚类可视化已保存为 '{clustering_file}'")
        except Exception as e:
            print(f"聚类可视化失败: {e}")
    
    return df, cluster_summary


# ============================================
# 第八部分：生成分析报告（支持批量）
# ============================================

def generate_analysis_report(df, topics_dict, vocab_stats, turning_points, 
                            cluster_summary, file_stats=None, file_paths=None):
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
                     'topic_confidence', 'cluster', 'global_line_number',
                     'file_name', 'file_path', 'file_id']]
    
    if metadata_cols:
        metadata_info = "\n数据集元数据:\n"
        for col in metadata_cols:
            if col in df.columns:
                metadata_info += f"  • {col}: {df[col].dtype}, {df[col].nunique()}个唯一值\n"
                if df[col].dtype == 'object' and df[col].nunique() < 10:
                    unique_vals = df[col].unique()
                    metadata_info += f"    值: {', '.join(map(str, unique_vals[:5]))}"
                    if len(unique_vals) > 5:
                        metadata_info += f", ... ({len(unique_vals)}个)"
                    metadata_info += "\n"
    
    # 生成报告文本
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
{'='*80}
动漫台词文本数据分析综合报告
{'='*80}

生成时间: {timestamp}
分析文件数: {df['file_name'].nunique() if 'file_name' in df.columns else 1}

一、数据集概览
{'─'*40}
• 总台词数: {len(df):,} 句
• 总文件数: {df['file_name'].nunique() if 'file_name' in df.columns else 1} 个
• 总词数: {vocab_stats['total_words']:,} 个
• 独特词汇: {vocab_stats['unique_words']:,} 个
• 词汇多样性: {vocab_stats['lexical_diversity']:.2%}
{metadata_info}

二、文件统计
{'─'*40}
"""
    
    # 添加文件统计信息
    if file_stats is not None:
        for i, (file_name, stats) in enumerate(file_stats.iterrows(), 1):
            short_name = file_name[:40] + "..." if len(file_name) > 40 else file_name
            report += f"{i}. {short_name}\n"
            report += f"   台词数: {stats['台词数']:,} | "
            report += f"平均词数: {stats['平均词数']:.1f} | "
            report += f"总词数: {stats['总词数']:,}\n"
    
    # 添加各文件情感对比
    if 'file_name' in df.columns and df['file_name'].nunique() > 1:
        report += f"\n三、各文件情感对比\n{'─'*40}\n"
        
        file_sentiment_summary = []
        for file_name in df['file_name'].unique():
            file_df = df[df['file_name'] == file_name]
            if len(file_df) > 0:
                positive_pct = (file_df['sentiment'] == 'positive').sum() / len(file_df)
                negative_pct = (file_df['sentiment'] == 'negative').sum() / len(file_df)
                neutral_pct = (file_df['sentiment'] == 'neutral').sum() / len(file_df)
                avg_sentiment = file_df['compound'].mean()
                
                file_sentiment_summary.append({
                    'file_name': file_name,
                    'positive': positive_pct,
                    'negative': negative_pct,
                    'neutral': neutral_pct,
                    'avg_sentiment': avg_sentiment
                })
        
        # 按平均情感排序
        file_sentiment_summary.sort(key=lambda x: x['avg_sentiment'], reverse=True)
        
        for i, file_summary in enumerate(file_sentiment_summary, 1):
            short_name = file_summary['file_name'][:30] + "..." if len(file_summary['file_name']) > 30 else file_summary['file_name']
            tone = "积极" if file_summary['avg_sentiment'] > 0.1 else "消极" if file_summary['avg_sentiment'] < -0.1 else "中性"
            report += f"{i}. {short_name:33s}: 积极{file_summary['positive']:5.1%} 消极{file_summary['negative']:5.1%} "
            report += f"中性{file_summary['neutral']:5.1%} | 平均情感: {file_summary['avg_sentiment']:.3f} ({tone})\n"
    
    report += f"\n四、整体台词长度分析\n{'─'*40}"
    report += f"""
• 平均字符长度: {length_stats['avg_chars']:.1f} (±{length_stats['std_chars']:.1f})
• 平均词数: {length_stats['avg_words']:.1f} (±{length_stats['std_words']:.1f})
• 最长台词: {length_stats['max_chars']} 字符 ({length_stats['max_words']} 词)
• 最短台词: {length_stats['min_chars']} 字符 ({length_stats['min_words']} 词)

五、整体情感分析结果
{'─'*40}
• 积极台词: {sentiment_stats['positive']:,} 句 ({sentiment_stats['positive_pct']:.1%})
• 消极台词: {sentiment_stats['negative']:,} 句 ({sentiment_stats['negative_pct']:.1%})
• 中性台词: {sentiment_stats['neutral']:,} 句 ({sentiment_stats['neutral_pct']:.1%})
• 平均情感强度: {sentiment_stats['avg_compound_abs']:.3f}
• 整体情感基调: {"积极向上" if sentiment_stats['avg_compound'] > 0.1 else "消极低沉" if sentiment_stats['avg_compound'] < -0.1 else "中性平衡"}
• 情感转折点: {len(turning_points)} 个

六、主题建模发现
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
        
        # 按文件统计主题分布
        file_topic_info = ""
        if 'file_name' in df.columns:
            file_topic_counts = df[df['dominant_topic'] == topic_id]['file_name'].value_counts()
            if len(file_topic_counts) > 0:
                top_files = file_topic_counts.head(3)
                file_topic_info = " | 主要文件: "
                file_topic_info += ", ".join([f"{file[:15]}...({count})" if len(file) > 15 else f"{file}({count})" 
                                            for file, count in top_files.items()])
        
        report += f"• {topic_name}: {', '.join(words[:8])}\n"
        report += f"  数量: {topic_count}句 ({topic_pct:.1%}) | "
        report += f"平均情感: {topic_sentiment:.3f} ({sentiment_label})"
        if file_topic_info:
            report += f"{file_topic_info}"
        report += "\n\n"
    
    # 添加聚类分析结果
    if cluster_summary:
        report += f"七、聚类分析结果\n{'─'*40}\n"
        report += f"共发现 {len(cluster_summary)} 个聚类:\n\n"
        
        for cluster in cluster_summary:
            report += f"• 聚类 {cluster['cluster_id']}:\n"
            report += f"  大小: {cluster['size']}句 ({cluster['size_pct']:.1%})\n"
            report += f"  平均情感: {cluster['avg_sentiment']:.3f}\n"
            if cluster['dominant_topic'] != -1:
                report += f"  主要主题: 主题 {cluster['dominant_topic'] + 1} ({cluster['topic_pct']:.1%})\n"
            
            # 显示文件分布
            if cluster['file_distribution']:
                report += f"  文件分布: "
                top_files = sorted(cluster['file_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
                file_strs = []
                for file_name, pct in top_files:
                    short_name = file_name[:15] + "..." if len(file_name) > 15 else file_name
                    file_strs.append(f"{short_name}({pct:.0%})")
                report += ", ".join(file_strs) + "\n"
            
            report += f"  代表性台词 ({cluster['example_file']}): {cluster['example']}\n\n"
    
    # 添加高频词汇
    report += f"八、高频词汇分析\n{'─'*40}\n"
    report += "前15个高频内容词（排除停用词）:\n\n"
    
    if vocab_stats['top_content_words']:
        for i, (word, freq) in enumerate(list(vocab_stats['top_content_words'].items())[:15], 1):
            freq_pct = freq / vocab_stats['total_words'] if vocab_stats['total_words'] > 0 else 0
            report += f"{i:2d}. {word:15s}: {freq:5d}次 ({freq_pct:.2%})\n"
    else:
        report += "无足够的高频词数据\n"
    
    # 添加各文件词汇多样性对比
    if vocab_stats['by_file'] is not None:
        report += f"\n九、各文件词汇多样性对比\n{'─'*40}\n"
        
        # 按词汇多样性排序
        file_diversity = []
        for file_name, stats in vocab_stats['by_file'].items():
            file_diversity.append((file_name, stats['lexical_diversity'], stats['total_words']))
        
        file_diversity.sort(key=lambda x: x[1], reverse=True)
        
        for i, (file_name, diversity, total_words) in enumerate(file_diversity, 1):
            short_name = file_name[:35] + "..." if len(file_name) > 35 else file_name
            report += f"{i:2d}. {short_name:38s}: 词汇多样性={diversity:.3%} (总词数={total_words:,})\n"
    
    # 添加叙事洞察
    report += f"\n十、叙事洞察\n{'─'*40}\n"
    
    # 分析情感变化模式
    if len(turning_points) > 0:
        report += "• 情感叙事有明显起伏，共发现 {} 个情感转折点\n".format(len(turning_points))
        
        # 按文件分析情感转折点
        if 'file_name' in df.columns:
            file_turning_points = {}
            for point_type, idx, value, line_text, file_name in turning_points:
                if file_name not in file_turning_points:
                    file_turning_points[file_name] = 0
                file_turning_points[file_name] += 1
            
            if file_turning_points:
                report += "• 各文件情感转折点数量:\n"
                for file_name, count in sorted(file_turning_points.items(), key=lambda x: x[1], reverse=True):
                    short_name = file_name[:25] + "..." if len(file_name) > 25 else file_name
                    report += f"  {short_name}: {count}个\n"
    
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
        report += f"\n十一、主要情感转折点\n{'─'*40}\n"
        for i, (point_type, idx, value, line_text, file_name) in enumerate(turning_points[:10], 1):
            point_name = "高潮" if point_type == 'peak' else "低谷"
            short_file = file_name[:20] + "..." if len(file_name) > 20 else file_name
            report += f"{i}. 第{idx+1}句 ({short_file}): {point_name} (情感强度: {value:.3f})\n"
            report += f"   台词: {line_text}...\n\n"
    
    # 添加分析建议
    report += f"\n十二、分析建议\n{'─'*40}\n"
    report += "1. 文件对比: 分析不同动漫作品的台词风格差异\n"
    report += "2. 角色识别: 使用命名实体识别自动识别说话者\n"
    report += "3. 上下文分析: 考虑前后台词的情感传递关系\n"
    report += "4. 文化语境: 添加更多日语动漫特有词汇到情感词典\n"
    report += "5. 叙事结构: 结合经典叙事理论分析起承转合\n"
    report += "6. 批量处理: 扩展支持更多文件格式和批量分析\n"
    
    # 添加技术说明
    report += f"\n十三、技术说明\n{'─'*40}\n"
    report += "• 情感分析: 使用NLTK的VADER，扩展了动漫相关词汇\n"
    report += "• 主题建模: 使用LDA算法，TF-IDF向量化\n"
    report += "• 聚类分析: 使用K-Means算法，轮廓系数选择最佳K值\n"
    report += "• 可视化: 使用Matplotlib和Seaborn创建综合仪表板\n"
    report += "• 文件支持: 支持.txt, .csv, .xlsx, .xls格式，支持批量分析\n"
    
    report += f"\n{'='*80}\n分析完成！\n{'='*80}"
    
    # 保存报告
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'anime_dialogue_analysis_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分析报告已保存为 '{report_file}'")
    except Exception as e:
        print(f"保存报告文件失败: {e}")
    
    # 在控制台显示报告摘要
    print("\n" + "="*60)
    print("报告摘要:")
    print(f"• 分析文件数: {df['file_name'].nunique() if 'file_name' in df.columns else 1}")
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
# 第九部分：导出结果（支持批量）
# ============================================

def export_results(df, topics_dict, file_paths=None):
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
    
    # 选择要导出的列
    base_columns = ['file_name', 'global_line_number', 'line_number', 
                   'cleaned_line', 'char_length', 'word_count',
                   'sentiment', 'compound', 'positive', 'negative', 
                   'neutral', 'dominant_topic', 'topic_keywords',
                   'topic_confidence']
    
    if 'cluster' in export_df.columns:
        base_columns.append('cluster')
    
    # 添加其他元数据列（如果有）
    extra_cols = [col for col in export_df.columns if col not in base_columns and 
                 col not in ['raw_line', 'file_path', 'file_id', 'compound_ma', 'processed_text']]
    
    export_columns = base_columns + extra_cols
    
    # 确保所有列都存在
    export_columns = [col for col in export_columns if col in export_df.columns]
    
    export_df = export_df[export_columns]
    
    # 保存为CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'anime_dialogue_analysis_results_{timestamp}.csv'
    
    try:
        export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"分析结果已保存为 '{csv_path}'")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")
    
    # 导出主题详情
    try:
        topics_df = pd.DataFrame(topics_dict)
        topics_path = f'anime_dialogue_topics_{timestamp}.csv'
        topics_df.to_csv(topics_path, encoding='utf-8-sig')
        print(f"主题详情已保存为 '{topics_path}'")
    except Exception as e:
        print(f"保存主题详情失败: {e}")
    
    # 导出按文件汇总的统计
    if 'file_name' in df.columns:
        try:
            # 文件级汇总统计
            file_summary = df.groupby('file_name').agg({
                'global_line_number': 'count',
                'word_count': ['mean', 'sum'],
                'char_length': ['mean', 'max'],
                'compound': ['mean', 'std'],
                'sentiment': lambda x: (x == 'positive').mean()
            }).round(3)
            
            # 重命名列
            file_summary.columns = ['台词数', '平均词数', '总词数', '平均字符数', 
                                   '最大字符数', '平均情感', '情感标准差', '积极比例']
            
            # 保存文件汇总
            file_summary_path = f'file_summary_{timestamp}.csv'
            file_summary.to_csv(file_summary_path, encoding='utf-8-sig')
            print(f"文件汇总统计已保存为 '{file_summary_path}'")
        except Exception as e:
            print(f"保存文件汇总统计失败: {e}")
    
    return export_df


# ============================================
# 第十部分：主函数（支持批量分析）
# ============================================

def main():
    """
    主函数：执行完整分析流程
    """
    print("="*60)
    print("动漫台词文本数据分析系统 - 批量分析版")
    print("="*60)
    
    # 配置参数
    # ============================================
    # 请根据您的需求修改以下配置：
    # ============================================
    
    # 方式1：指定文件路径列表（推荐）
    file_paths = [
        r"D:\python\风之谷\副本风之谷.xlsx",
        r"D:\python\哈尔的移动城堡\副本哈尔的移动城堡.xlsx",
        r"D:\python\红猪\副本红猪.xlsx",
        r"D:\python\借东西的小人阿莉埃蒂\副本借东西的小人阿莉埃蒂.xlsx",
        r"D:\python\龙猫\副本龙猫.xlsx",
        r"D:\python\魔女宅急便\副本魔女宅急便.xlsx",
        r"D:\python\起风了\副本起风了.xlsx",
        r"D:\python\千与千寻\副本千与千寻.xlsx",
        r"D:\python\天空之城\副本天空之城.xlsx",
        r"D:\python\崖上的金姬鱼\副本崖上的金姬鱼.xlsx",
        r"D:\python\幽灵公主\副本幽灵公主.xlsx",
        r"D:\python\你想活出怎样的人生\副本你想活出怎样的人生.xlsx"
    ]
    
    # 方式2：使用通配符批量加载文件
    # file_pattern = r"D:\python\*.xlsx"  # 加载所有Excel文件
    # file_paths = glob.glob(file_pattern)
    
    # Excel文件中台词文本所在的列名
    text_column = "cueline"  # 如果您的Excel中台词列不是'cueline'，请修改
    
    # 是否执行聚类分析（数据量小时可以关闭）
    perform_clustering = True
    
    # ============================================
    # 以下为执行代码，一般不需要修改
    # ============================================
    
    try:
        # 检查文件路径
        if not file_paths:
            print("错误: 没有指定文件路径")
            print("请修改main()函数中的file_paths变量")
            return
        
        # 验证文件是否存在
        valid_files = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                print(f"警告: 文件不存在: {file_path}")
        
        if not valid_files:
            print("错误: 没有有效的文件路径")
            return
        
        print(f"准备分析 {len(valid_files)} 个文件:")
        for i, file_path in enumerate(valid_files, 1):
            print(f"  {i}. {os.path.basename(file_path)}")
        
        # 步骤1: 批量加载和清洗数据
        print("\n步骤1: 批量加载和清洗数据...")
        df, file_stats = load_multiple_files(valid_files, text_column)
        
        if df is None or len(df) == 0:
            print("数据加载失败，请检查文件路径和格式")
            return
        
        if len(df) < 10:
            print("警告: 数据量较少，分析结果可能不准确")
            print("建议至少提供50句台词以获得更好的分析结果")
        
        print(f"成功加载 {len(df)} 句台词，来自 {df['file_name'].nunique()} 个文件")
        
        # 步骤2: 词汇统计分析
        vocab_stats = analyze_vocabulary(df, by_file=True)
        
        # 步骤3: 情感分析
        df, turning_points = perform_sentiment_analysis(df)
        
        # 步骤4: 主题建模
        df, topics_dict, lda_model, X_matrix, vectorizer = perform_topic_modeling(df)
        
        # 步骤5: 可视化分析
        fig = create_visualizations(df, topics_dict, turning_points, vocab_stats, file_stats)
        
        # 步骤6: 聚类分析（可选，数据量足够时执行）
        if perform_clustering and len(df) > 30:
            topic_distribution = lda_model.transform(
                vectorizer.transform(df['processed_text'])
            )
            df, cluster_summary = perform_clustering_analysis(df, topic_distribution)
        else:
            print("\n跳过聚类分析（数据量较少或已禁用）")
            cluster_summary = []
        
        # 步骤7: 生成分析报告
        report = generate_analysis_report(
            df, topics_dict, vocab_stats, turning_points, 
            cluster_summary, file_stats, valid_files
        )
        
        # 步骤8: 导出结果
        export_df = export_results(df, topics_dict, valid_files)
        
        print("\n" + "="*60)
        print("批量分析完成！")
        print("生成的文件:")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"  1. anime_dialogue_analysis_{timestamp}.png - 可视化图表")
        print(f"  2. anime_dialogue_analysis_report_{timestamp}.txt - 详细分析报告")
        print(f"  3. anime_dialogue_analysis_results_{timestamp}.csv - 分析结果数据")
        print(f"  4. anime_dialogue_topics_{timestamp}.csv - 主题详情")
        print(f"  5. file_summary_{timestamp}.csv - 文件汇总统计")
        
        if perform_clustering and len(df) > 30 and cluster_summary:
            print(f"  6. clustering_visualization_{timestamp}.png - 聚类可视化")
        
        print("="*60)
        
        # 显示一些示例分析结果
        print("\n示例分析结果:")
        if len(df) > 0:
            # 按文件显示代表性结果
            if 'file_name' in df.columns:
                for file_name in df['file_name'].unique()[:3]:  # 只显示前3个文件
                    file_df = df[df['file_name'] == file_name]
                    if len(file_df) > 0:
                        short_name = file_name[:30] + "..." if len(file_name) > 30 else file_name
                        print(f"\n文件: {short_name}")
                        
                        # 最积极台词
                        if file_df['compound'].max() > 0:
                            most_positive = file_df.loc[file_df['compound'].idxmax()]
                            print(f"  最积极台词: '{most_positive['cleaned_line'][:80]}...'")
                        
                        # 最消极台词
                        if file_df['compound'].min() < 0:
                            most_negative = file_df.loc[file_df['compound'].idxmin()]
                            print(f"  最消极台词: '{most_negative['cleaned_line'][:80]}...'")
                        
                        # 平均情感
                        avg_sentiment = file_df['compound'].mean()
                        tone = "积极" if avg_sentiment > 0.1 else "消极" if avg_sentiment < -0.1 else "中性"
                        print(f"  平均情感: {avg_sentiment:.3f} ({tone})")
            
            # 整体代表性结果
            print("\n整体分析:")
            
            # 最长台词
            longest_idx = df['char_length'].idxmax()
            longest_file = df.loc[longest_idx, 'file_name'] if 'file_name' in df.columns else "整体"
            print(f"  最长台词 ({longest_file}):")
            print(f"    '{df.loc[longest_idx, 'cleaned_line'][:80]}...'")
            
            # 最积极台词
            if df['compound'].max() > 0:
                most_positive_idx = df['compound'].idxmax()
                positive_file = df.loc[most_positive_idx, 'file_name'] if 'file_name' in df.columns else "整体"
                print(f"\n  最积极台词 ({positive_file}):")
                print(f"    '{df.loc[most_positive_idx, 'cleaned_line'][:80]}...'")
            
            # 最消极台词
            if df['compound'].min() < 0:
                most_negative_idx = df['compound'].idxmin()
                negative_file = df.loc[most_negative_idx, 'file_name'] if 'file_name' in df.columns else "整体"
                print(f"\n  最消极台词 ({negative_file}):")
                print(f"    '{df.loc[most_negative_idx, 'cleaned_line'][:80]}...'")
        
    except FileNotFoundError as e:
        print(f"错误: 找不到文件: {e}")
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
    print("2. 修改main()函数中的file_paths变量为您的文件路径列表")
    print("3. 如果需要，修改text_column变量为您的台词列名")
    print("4. 运行此脚本")
    print("5. 查看生成的分析文件和报告")

import sys
import re
import jieba
from collections import defaultdict
import logging
from typing import List, Dict, Set

# 配置日志
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)

# 预编译正则表达式（一次编译多次使用）
PUNCTUATION_PATTERN = re.compile(r'[^\w\s\u4e00-\u9fa5]')
WHITESPACE_PATTERN = re.compile(r'\s+')

# 预加载停用词集合（高频出现的无意义词汇）
STOPWORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
             '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
             '没有', '看', '好', '自己', '这', '也', '但', '而', '于', '之', '以'}


def preprocess_text(text: str) -> List[str]:
    #文本预处理优化
    if not text:
        return []

    # 合并字符串处理步骤，减少中间变量
    text = PUNCTUATION_PATTERN.sub('', text)  # 移除标点
    text = WHITESPACE_PATTERN.sub(' ', text).strip()  # 处理空格

    words = jieba.lcut(text, cut_all=False, HMM=False)

    # 单次循环完成过滤，避免额外函数调用开销
    filtered = []
    for word in words:
        if len(word) > 1 and word not in STOPWORDS:
            filtered.append(word)
    return filtered


def build_fast_index(words: List[str]) -> tuple[Set[str], Dict[str, int], int]:
    word_set = set()
    word_freq = dict()  # 使用普通dict而非defaultdict，减少属性查找开销
    total = 0

    for word in words:
        word_set.add(word)
        word_freq[word] = word_freq.get(word, 0) + 1
        total += 1

    return word_set, word_freq, total


def calculate_fast_similarity(original_set: Set[str],
                              original_freq: Dict[str, int],
                              plagiarized_words: List[str]) -> float:
    #快速相似度计算：减少数学运算，合并循环
    len_plag = len(plagiarized_words)
    if not original_set or len_plag == 0:
        return 0.0

    # 合并所有计算到单个循环中，减少遍历次数
    matched = 0
    dot_product = 0
    orig_sq_sum = 0
    plag_sq_sum = 0
    plag_freq = dict()

    # 一次遍历完成抄袭文的所有必要计算
    for word in plagiarized_words:
        # 统计词频
        plag_freq[word] = plag_freq.get(word, 0) + 1

        # 检查是否在原文中
        if word in original_set:
            matched += 1
            # 预取原文词频，减少字典查找
            orig_count = original_freq[word]
            dot_product += orig_count * plag_freq[word]
            orig_sq_sum += orig_count ** 2
            plag_sq_sum += plag_freq[word] ** 2

    # 快速计算匹配率
    match_ratio = matched / len_plag

    # 无匹配直接返回
    if matched == 0:
        return 0.0

    # 简化余弦相似度计算
    if orig_sq_sum == 0 or plag_sq_sum == 0:
        cosine = 0.0
    else:
        # 合并计算步骤，减少中间变量
        cosine = dot_product / ((orig_sq_sum ** 0.5) * (plag_sq_sum ** 0.5))

    # 简化加权计算
    return (cosine * 0.7 + match_ratio * 0.3)


def main():
    try:
        # 快速参数检查
        if len(sys.argv) != 4:
            print("用法: python ultrafast_plagiarism_checker.py <原文路径> <抄袭版路径> <输出路径>")
            sys.exit(1)

        original_path, plagiarized_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

        # 最快方式读取文件（减少I/O操作）
        with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_text = f.read()

        with open(plagiarized_path, 'r', encoding='utf-8', errors='ignore') as f:
            plagiarized_text = f.read()

        # 预处理与索引构建
        original_words = preprocess_text(original_text)
        original_set, original_freq, _ = build_fast_index(original_words)

        # 处理抄袭文本
        plagiarized_words = preprocess_text(plagiarized_text)

        # 计算相似度
        similarity = calculate_fast_similarity(original_set, original_freq, plagiarized_words)

        # 快速写入结果
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")

    except Exception as e:
        logger.error(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 禁用jieba的所有日志输出
    jieba.setLogLevel(logging.ERROR)
    # 预加载jieba词典以加快首次分词速度
    jieba.initialize()
    main()

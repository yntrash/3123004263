import unittest
import os
import tempfile
import sys
import subprocess

# 将当前目录添加到Python路径，确保能导入主程序
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入主程序函数
from main import (
    preprocess_text,
    build_fast_index,
    calculate_fast_similarity
)


class TestPlagiarismChecker(unittest.TestCase):
    # 测试用例数据
    TEST_ORIGINAL = "今天是星期天，天气晴，今天晚上我要去看电影。"
    TEST_PLAGIARIZED = "今天是周天，天气晴朗，我晚上要去看电影。"
    TEST_EMPTY = ""
    TEST_SPECIAL_CHARS = "论文标题：基于Python的查重算法研究\n关键词：查重；Python；算法\n摘要：本文介绍了1种基于倒排索引的查重方法。"

    def test_preprocess_text_basic(self):
        result = preprocess_text(self.TEST_ORIGINAL)
        self.assertTrue(len(result) > 0)
        self.assertIn("今天", result)
        self.assertIn("天气", result)
        self.assertIn("电影", result)
        self.assertNotIn("是", result)  # 停用词应被过滤
        self.assertNotIn("，", result)  # 标点应被过滤

    def test_preprocess_text_empty(self):
        #测试空文本和特殊字符处理
        self.assertEqual(preprocess_text(self.TEST_EMPTY), [])
        self.assertEqual(preprocess_text("   "), [])
        self.assertEqual(preprocess_text("！@#￥%……&*"), [])

    def test_preprocess_text_special_chars(self):
        #测试含特殊字符的文本预处理
        result = preprocess_text(self.TEST_SPECIAL_CHARS)
        expected_words = [
            "论文", "标题", "基于", "Python", "算法", "研究",
            "关键词", "Python", "算法", "摘要", "本文", "介绍",
            "基于", "倒排", "索引", "方法"
        ]
        for word in expected_words:
            self.assertIn(word, result)

    def test_build_fast_index(self):
        #测试索引构建功能
        words = ["今天", "天气", "今天", "晚上", "电影"]
        word_set, word_freq, total = build_fast_index(words)

        self.assertEqual(word_set, {"今天", "天气", "晚上", "电影"})
        self.assertEqual(word_freq["今天"], 2)
        self.assertEqual(word_freq["天气"], 1)
        self.assertEqual(total, 5)

    def test_build_fast_index_empty(self):
        #测试空列表的索引构建
        word_set, word_freq, total = build_fast_index([])
        self.assertEqual(word_set, set())
        self.assertEqual(word_freq, {})
        self.assertEqual(total, 0)

    def test_calculate_similarity_identical(self):
        #测试完全相同文本的相似度
        original_words = ["今天", "天气", "晚上", "电影"]
        plagiarized_words = ["今天", "天气", "晚上", "电影"]

        orig_set, orig_freq, _ = build_fast_index(original_words)
        similarity = calculate_fast_similarity(orig_set, orig_freq, plagiarized_words)

        self.assertAlmostEqual(similarity, 1.0, places=1)

    def test_calculate_similarity_none(self):
        #测试完全不同文本的相似度
        original_words = ["苹果", "香蕉", "橙子"]
        plagiarized_words = ["汽车", "火车", "飞机"]

        orig_set, orig_freq, _ = build_fast_index(original_words)
        similarity = calculate_fast_similarity(orig_set, orig_freq, plagiarized_words)

        self.assertEqual(similarity, 0.0)

    def test_calculate_similarity_partial(self):
        #测试部分相似文本的相似度
        original_words = ["今天", "星期天", "天气", "晚上", "看", "电影"]
        plagiarized_words = ["今天", "周天", "天气", "晚上", "看", "电影"]

        orig_set, orig_freq, _ = build_fast_index(original_words)
        similarity = calculate_fast_similarity(orig_set, orig_freq, plagiarized_words)

        self.assertGreater(similarity, 0.8)
        self.assertLess(similarity, 1.0)

    def test_calculate_similarity_subset(self):
        #测试抄袭文本是原文子集的情况
        original_words = ["第一章", "介绍", "研究", "背景", "目的", "意义"]
        plagiarized_words = ["介绍", "研究", "目的"]

        orig_set, orig_freq, _ = build_fast_index(original_words)
        similarity = calculate_fast_similarity(orig_set, orig_freq, plagiarized_words)

        self.assertGreater(similarity, 0.8)

    def test_full_file_processing(self):
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as orig_file:
            orig_file.write(self.TEST_ORIGINAL)
            orig_path = orig_file.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as plag_file:
            plag_file.write(self.TEST_PLAGIARIZED)
            plag_path = plag_file.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as output_file:
            output_path = output_file.name

        try:
            # 运行主程序
            result = subprocess.run(
                [sys.executable, "main.py", orig_path, plag_path, output_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # 验证程序正常运行
            self.assertEqual(result.returncode, 0, f"主程序运行错误: {result.stderr}")

            # 验证输出结果
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.assertRegex(content, r'^\d+\.\d{2}$')

                similarity = float(content)
                self.assertGreater(similarity, 0.6)
                self.assertLess(similarity, 1.0)

        finally:
            # 清理临时文件
            for path in [orig_path, plag_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)

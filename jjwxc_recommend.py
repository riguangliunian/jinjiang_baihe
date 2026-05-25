"""
晋江文学城百合小说推荐系统
基于 tag + 简介 的混合推荐，支持"输入书名"或"输入想看类型"两种模式
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------- 分词（纯 Python，无外部依赖） ----------

STOPWORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 这 中 大 为 上 个 国 们 到 说 时 "
    "地 要 也 子 里 去 之 会 着 没有 看 好 自己 这 她 他 它 被 从 那 你 "
    "以 但 最 又 很 与 及 等 还 把 可 能 对 而 让 所 之 其 如 何 已 "
    "什么 怎么 这个 那个 如果 因为 所以 但是 而且 或者 可以 已经 "
    "他们 她们 我们 你们 不是 没有 一个 一些 一样 一直 一种 下来 "
    "出来 起来 过来 回来 出来 只有 只是 就是 还是 只是 不过".split()
)


def tokenize(text: str) -> list[str]:
    """简易中文分词：按标点切短句，再按2-4字滑动窗口 + 单字"""
    if not text:
        return []
    chunks = re.split(r"[，。！？、；：""''\s\-\—\(\)（）\[\]【】…\n\r\t]+", text)
    tokens = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # 2-4 字 n-gram
        for n in (2, 3, 4):
            for i in range(len(chunk) - n + 1):
                w = chunk[i : i + n]
                tokens.append(w)
        # 单字（过滤停用词和短词）
        for ch in chunk:
            if ch not in STOPWORDS and len(ch.strip()) > 0:
                tokens.append(ch)
    return tokens


# ---------- TF-IDF ----------


class TfidfEngine:
    def __init__(self, documents: list[list[str]]):
        self.n_docs = len(documents)
        # document frequency
        df = Counter()
        for doc in documents:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: math.log((self.n_docs + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

    def tfidf(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {
            term: (count / total) * self.idf.get(term, 1.0)
            for term, count in tf.items()
        }


def cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in v1.values()))
    norm2 = math.sqrt(sum(v * v for v in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def jaccard_sim(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# ---------- 推荐系统 ----------


class Recommender:
    # 权重：tag 相似度 vs 简介相似度
    W_TAG = 0.45
    W_SYN = 0.55

    def __init__(self, data_path: str = "jjwxc_baihe.json"):
        self.data_path = Path(data_path)
        self.books: list[dict] = []
        self.tag_sets: list[set[str]] = []
        self.syn_tokens: list[list[str]] = []
        self.syn_vectors: list[dict[str, float]] = []
        self.tfidf_engine: TfidfEngine | None = None
        self.title_index: dict[str, int] = {}  # 书名 -> index
        self._load()

    def _load(self):
        if not self.data_path.exists():
            print(f"数据文件不存在: {self.data_path}")
            print("请先运行 jjwxc_baihe.py 爬取数据")
            sys.exit(1)

        with open(self.data_path, encoding="utf-8") as f:
            self.books = json.load(f)

        print(f"已加载 {len(self.books)} 本书\n")

        # 解析 tag
        for book in self.books:
            raw = book.get("tags", "")
            tags = {t.strip() for t in raw.split("、") if t.strip()}
            self.tag_sets.append(tags)

        # 分词
        for book in self.books:
            tokens = tokenize(book.get("synopsis", ""))
            self.syn_tokens.append(tokens)

        # 构建 TF-IDF
        self.tfidf_engine = TfidfEngine(self.syn_tokens)
        self.syn_vectors = [self.tfidf_engine.tfidf(t) for t in self.syn_tokens]

        # 书名索引（模糊匹配用小写）
        for i, book in enumerate(self.books):
            self.title_index[book.get("title", "").lower()] = i

    def _find_book(self, query: str) -> int | None:
        """模糊匹配书名"""
        q = query.lower().strip()
        if q in self.title_index:
            return self.title_index[q]
        # 部分匹配
        for title, idx in self.title_index.items():
            if q in title or title in q:
                return idx
        return None

    def recommend_by_book(self, book_name: str, top_n: int = 10) -> list[dict]:
        """根据书名推荐相似书籍"""
        idx = self._find_book(book_name)
        if idx is None:
            print(f"未找到《{book_name}》，尝试按类型推荐...")
            return self.recommend_by_text(book_name, top_n)
        return self._rank(idx=idx, top_n=top_n)

    def recommend_by_text(self, text: str, top_n: int = 10) -> list[dict]:
        """根据用户输入的描述/类型推荐"""
        # 把输入当作"虚拟 tag + 虚拟简介"
        # 先尝试按顿号/逗号提取 tag
        user_tags = set()
        for sep in ("、", ",", "，"):
            for part in text.split(sep):
                part = part.strip()
                if part and len(part) <= 8:
                    user_tags.add(part)
        # 没提取出 tag 就整体作为一个 tag
        if not user_tags:
            user_tags = {text.strip()}

        user_syn_tokens = tokenize(text)
        user_syn_vec = self.tfidf_engine.tfidf(user_syn_tokens)

        scores = []
        for i in range(len(self.books)):
            tag_s = jaccard_sim(user_tags, self.tag_sets[i])
            syn_s = cosine_sim(user_syn_vec, self.syn_vectors[i])
            score = self.W_TAG * tag_s + self.W_SYN * syn_s
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_n]:
            if score > 0:
                results.append(self._format(idx, score))
        return results

    def _rank(self, idx: int, top_n: int = 10) -> list[dict]:
        """给定一本书的 index，计算与所有书的相似度"""
        tag_set = self.tag_sets[idx]
        syn_vec = self.syn_vectors[idx]

        scores = []
        for i in range(len(self.books)):
            if i == idx:
                continue
            tag_s = jaccard_sim(tag_set, self.tag_sets[i])
            syn_s = cosine_sim(syn_vec, self.syn_vectors[i])
            score = self.W_TAG * tag_s + self.W_SYN * syn_s
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_n]:
            results.append(self._format(i, score))
        return results

    def _format(self, idx: int, score: float) -> dict:
        book = self.books[idx]
        return {
            "title": book.get("title", ""),
            "author": book.get("author", ""),
            "tags": book.get("tags", ""),
            "synopsis": (book.get("synopsis", "")[:200] + "...")
            if len(book.get("synopsis", "")) > 200
            else book.get("synopsis", ""),
            "score": round(score, 4),
        }


# ---------- 交互 ----------


def main():
    print("=" * 50)
    print("  晋江百合小说推荐系统")
    print("=" * 50)
    print()
    print("两种使用方式：")
    print("  1. 输入书名    → 推荐类似的书")
    print("  2. 输入想看类型 → 推荐 matching 的书")
    print("  输入 q 退出\n")

    rec = Recommender()

    while True:
        query = input("🔍 请输入书名或想看的类型: ").strip()
        if query.lower() in ("q", "quit", "exit", "退出"):
            print("再见！")
            break
        if not query:
            continue

        # 判断是书名还是描述（简单启发：如果匹配到书名就按书名推荐）
        idx = rec._find_book(query)
        if idx is not None:
            print(f"\n📖 找到《{rec.books[idx]['title']}》，为您推荐相似书籍：\n")
            results = rec.recommend_by_book(query, top_n=10)
        else:
            print(f"\n📖 根据您的偏好「{query}」，为您推荐：\n")
            results = rec.recommend_by_text(query, top_n=10)

        if not results:
            print("  没有找到匹配的书籍，换个关键词试试？\n")
            continue

        for rank, item in enumerate(results, 1):
            print(f"  {rank:>2}. 《{item['title']}》 by {item['author']}")
            print(f"      标签: {item['tags']}")
            print(f"      简介: {item['synopsis']}")
            print(f"      匹配度: {item['score']}")
            print()

        print("-" * 50)


if __name__ == "__main__":
    main()
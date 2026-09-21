#!/usr/bin/env python3
"""对 Seedance 提示词执行可重复的通用交付校验。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any


# 这些表达会把提示词的执行条件绑定到聊天、上一镜或前一次回答，而不是本次正文与提交素材。
EXTERNAL_CONTEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"承接\s*(?:上|前)一镜(?:头)?", "请直接写出本镜开场状态，或明确绑定到已提交的 @视频N。"),
    (r"(?<!承)(?:接|衔接|沿用)\s*(?:上|前)一镜(?:头)?", "请直接写出需要继承的状态，或明确绑定到已提交的 @视频N。"),
    (r"(?:上|前)一镜(?:头)?(?:中|里|的)", "请把所指的人物、位置、动作或构图写入本次正文。"),
    (r"(?:同前|如前)(?![述文])", "请展开为本次可执行的具体要求。"),
    (r"此前(?:的)?(?:位置|状态|构图|动作|造型|场景)", "请在正文中明确该位置或状态。"),
    (r"(?:你|我)?刚才(?:说|提|给|写|生成|描述|确认)?(?:的|过)?", "请把依赖的内容直接写入提示词。"),
    (r"(?:本次|当前)?(?:聊天|对话)(?:中|里|前文|上文)?", "提示词不能依赖聊天记录，请写出完整事实。"),
    (r"(?:前述|上述|前面提到的|之前提到的)(?:位置|状态|要求|内容|设定|人物|场景|动作)?", "请展开为本次正文中的明确要求。"),
    (r"(?:沿用|保持)(?:之前|此前)(?:的)?", "请明确列出需要继承的属性。"),
)

# 这些词常暗含未写出的前置状态；只有正文已给出起点或明确引用提交素材时才可安全执行。
AMBIGUOUS_CONTINUATION_WORDS: tuple[str, ...] = ("仍", "继续", "恢复", "重新")

# 能在同一提示词内建立明确起点的常见表达，不限定具体题材、主体或镜头内容。
LOCAL_START_PATTERNS: tuple[str, ...] = (
    r"镜头(?:开始|开场)时",
    r"(?:开场|起始|一开始)(?:时|画面)?",
    r"初始状态",
    r"起点(?:是|为|位于)",
    r"已经[^，。；;\n]{1,40}",
    r"从[^，。；;\n]{1,40}(?:开始|起步|出发|延续)",
)

# 这些内容通常来自交付说明、门禁报告或修复过程，不应被复制进模型提示词正文。
META_TEXT_PATTERNS: tuple[str, ...] = (
    "门禁通过",
    "门禁未通过",
    "校验通过",
    "校验结果",
    "检查结果",
    "修复说明",
    "修改说明",
    "原因推断",
    "已见事实",
    "交付判定",
    "以下是提示词",
    "最终提示词如下",
)

# 未完成模板、变量和假编号会让交付看似完整但无法直接提交。
PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"\b(?:TODO|TBD|FIXME)\b",
    r"(?:待填写|待补充|待确认|待替换|此处填写|此处补充)",
    r"\{\{[^{}\n]+\}\}",
    r"\$\{[^{}\n]+\}",
    r"@(?:图片|视频|音频)\s*[NXＸnｎ](?!\d)",
    r"(?:图片|视频|音频)\s*[XＸ](?!\d)",
    r"\[\s*(?:占位|待填|待补)[^\]]*\]",
)

# 2.0 使用不同括号绑定持续声场、动作音效、对白与画面文字。
SEEDANCE_20_LABELS: tuple[tuple[str, str, str, str], ...] = (
    ("持续声场", r"(?:持续声场|环境声|背景音乐|配乐|音乐)", "（", "）"),
    ("动作音效", r"(?:动作音效|音效)", "<", ">"),
    ("对白", r"(?:台词|对白)", "{", "}"),
    ("画面文字", r"(?:字幕|画面文字)", "【", "】"),
)

# 成对符号检查只覆盖 Seedance 标签相关符号，避免把自然语言标点误当成模板语法。
DELIMITER_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("（", "）", "持续声场"),
    ("<", ">", "动作音效"),
    ("{", "}", "对白"),
    ("【", "】", "画面文字"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，并把标准输入作为默认提示词来源。

    路径参数使用 ``-`` 表示标准输入；版本与任务类型用于启用对应规则；严格模式把警告也视为交付失败，
    JSON 模式则供其他脚本或自动化流程稳定读取校验结果。
    """

    parser = argparse.ArgumentParser(description="校验 Seedance 提示词是否可独立、清晰地交付。")
    parser.add_argument("path", nargs="?", default="-", help="提示词文件路径；省略或使用 - 时从标准输入读取。")
    parser.add_argument("--version", choices=("2.0", "2.5"), default="2.0", help="Seedance 版本，默认 2.0。")
    parser.add_argument(
        "--task",
        choices=("auto", "new", "edit", "extend", "bridge"),
        default="auto",
        help="任务类型；默认根据正文推断。",
    )
    parser.add_argument("--strict", action="store_true", help="把警告也作为失败，适合完整提示词交付门禁。")
    parser.add_argument("--json", action="store_true", dest="json_output", help="以 JSON 输出结果。")
    return parser.parse_args(argv)


def read_prompt(path_value: str) -> str:
    """从 UTF-8 文件或标准输入读取完整提示词。

    读取阶段不改写空白或标点，避免校验器在用户不知情时改变提示词。文件不存在、编码错误或输入为空等问题
    由调用方统一转换为可读错误和退出码 2。
    """

    if path_value == "-":
        return sys.stdin.read()
    return Path(path_value).read_text(encoding="utf-8")


def infer_task_type(prompt: str, requested_task: str) -> str:
    """在未显式指定时，根据任务动词和素材引用推断生成类型。

    推断只影响“是否允许绑定到已提交视频的续接表达”等校验，不替代模型或接口的真实任务参数。显式参数始终优先，
    避免校验器把用户已经确认的编辑、延长或桥接任务改判为新生成。
    """

    if requested_task != "auto":
        return requested_task
    if re.search(r"(?:桥接|过渡|衔接).{0,30}@视频\s*\d+.{0,80}@视频\s*\d+", prompt, re.S):
        return "bridge"
    if re.search(r"(?:延长|续写|续接|接续)\s*@视频\s*\d+|@视频\s*\d+.{0,20}(?:延长|续写|续接)", prompt):
        return "extend"
    if re.search(r"(?:编辑|修改|替换|移除|改变|调整)\s*@视频\s*\d+|@视频\s*\d+.{0,20}(?:编辑|修改|替换|移除|改变|调整)", prompt):
        return "edit"
    return "new"


def line_column(prompt: str, index: int) -> tuple[int, int]:
    """把字符偏移转换为从 1 开始的行列位置。

    诊断信息使用用户可直接定位的行号和列号；即使提示词只有一个长段，也能通过列号找到触发词。
    """

    line = prompt.count("\n", 0, index) + 1
    previous_newline = prompt.rfind("\n", 0, index)
    column = index - previous_newline
    return line, column


def make_issue(
    prompt: str,
    severity: str,
    code: str,
    message: str,
    index: int = 0,
    excerpt: str = "",
    suggestion: str = "",
) -> dict[str, Any]:
    """构造结构稳定的单条诊断记录。

    每条记录同时保留严重级别、机器码、自然语言说明、位置、触发片段和修复建议，既便于命令行阅读，
    也便于未来的编辑器、CI 或生成工作流消费 JSON，而不依赖输出文案解析。
    """

    line, column = line_column(prompt, max(index, 0))
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "line": line,
        "column": column,
        "excerpt": excerpt,
        "suggestion": suggestion,
    }


def sentence_span(prompt: str, index: int) -> tuple[int, int, str]:
    """取得触发位置所在的局部句子，用于判断素材绑定和展示诊断。

    句子边界覆盖常见中英文终止标点及换行；局部窗口可以防止某处出现 ``@视频1`` 就错误地豁免全文所有
    “上一镜”表达，从而保持素材绑定必须就近、明确。
    """

    left_matches = [prompt.rfind(mark, 0, index) for mark in ("。", "！", "？", ";", "；", "\n")]
    start = max(left_matches) + 1
    end_candidates = [position for mark in ("。", "！", "？", ";", "；", "\n") if (position := prompt.find(mark, index)) >= 0]
    end = min(end_candidates) if end_candidates else len(prompt)
    return start, end, prompt[start:end].strip()


def is_bound_to_submitted_video(sentence: str, task_type: str) -> bool:
    """判断续接语义是否明确绑定到本次提交的视频素材。

    只有编辑、延长或桥接任务，并且同一句实际出现 ``@视频N`` 时才允许依赖素材内部状态。新生成任务中的
    视频编号不能自动把“上一镜”变成可靠引用，避免任务类型混用。
    """

    return task_type in {"edit", "extend", "bridge"} and bool(re.search(r"@视频\s*\d+", sentence))


def check_external_context(prompt: str, task_type: str) -> list[dict[str, Any]]:
    """检查提示词是否依赖聊天、上一镜或前一次回答中的隐含事实。

    明确绑定到提交视频的编辑、延长和桥接语句可以使用素材自身的结尾状态；除此之外，所有起点、位置、造型、
    动作和构图都必须在本次正文或明确素材职责中给出。
    """

    issues: list[dict[str, Any]] = []
    for pattern, suggestion in EXTERNAL_CONTEXT_PATTERNS:
        for match in re.finditer(pattern, prompt):
            _, _, sentence = sentence_span(prompt, match.start())
            if is_bound_to_submitted_video(sentence, task_type):
                continue
            issues.append(
                make_issue(
                    prompt,
                    "error",
                    "external-context",
                    "提示词依赖未随本次请求提交的外部上下文。",
                    match.start(),
                    match.group(0),
                    suggestion,
                )
            )
    return issues


def has_local_start_state(prompt: str, index: int, end_index: int, sentence: str, task_type: str) -> bool:
    """判断含糊续接词之前是否已经存在本地可执行的起点。

    检查触发词前的正文、同一句明确开场标记以及已提交视频绑定。该方法只证明“存在先行状态”，不尝试理解
    所有剧情语义，因此命中时仍由语义镜头合同判断具体状态是否充分。
    """

    if is_bound_to_submitted_video(sentence, task_type):
        return True
    prefix = prompt[:index]
    local_window = prefix[-240:]
    if any(re.search(pattern, local_window) for pattern in LOCAL_START_PATTERNS):
        return True

    # “先……随后恢复”“松开后重新握住”等动作链已经在正文内给出前因，不应被当作聊天外依赖。
    if re.search(r"(?:先|先是|一度|短暂|随后|然后|接着|片刻后|数秒后|之后|以后)[^。！？!?\n]{0,80}$", local_window):
        return True

    # 若续接词后的核心短语已在前文出现，也可确认其先行状态来自本次正文。
    continuation = prompt[end_index:]
    continuation = re.split(r"[，,。；;！!？?]", continuation, maxsplit=1)[0]
    continuation = re.sub(r"[^\w\u4e00-\u9fff]+", "", continuation)
    if len(continuation) >= 4 and continuation[:4] in re.sub(r"[^\w\u4e00-\u9fff]+", "", local_window):
        return True
    return False


def check_ambiguous_continuations(prompt: str, task_type: str) -> list[dict[str, Any]]:
    """检查没有本地先行状态的“仍、继续、恢复、重新”等续接词。

    这类词不一定错误，因此默认产生警告；完整提示词交付使用 ``--strict`` 后会阻断。若正文先写明起始状态，
    或编辑、延长任务就近绑定 ``@视频N``，则不报告。
    """

    issues: list[dict[str, Any]] = []
    for word in AMBIGUOUS_CONTINUATION_WORDS:
        for match in re.finditer(re.escape(word), prompt):
            _, _, sentence = sentence_span(prompt, match.start())
            if has_local_start_state(prompt, match.start(), match.end(), sentence, task_type):
                continue
            issues.append(
                make_issue(
                    prompt,
                    "warning",
                    "ambiguous-continuation",
                    f"“{word}”可能依赖未写出的先行状态。",
                    match.start(),
                    sentence[:80],
                    "先明确本次正文中的开场或前置动作；若来自素材，请就近绑定到 @视频N。",
                )
            )
    return issues


def check_placeholders(prompt: str) -> list[dict[str, Any]]:
    """检查未替换的模板变量、待办标记和假素材编号。

    校验器只识别明显占位形式，不把真实的 ``@图片1``、``@视频2`` 或自然语言中的普通方括号误判为占位符。
    """

    issues: list[dict[str, Any]] = []
    for pattern in PLACEHOLDER_PATTERNS:
        for match in re.finditer(pattern, prompt, re.I):
            issues.append(
                make_issue(
                    prompt,
                    "error",
                    "unfinished-placeholder",
                    "提示词包含未完成的占位内容。",
                    match.start(),
                    match.group(0),
                    "填写实际内容或删除该占位符后再交付。",
                )
            )
    return issues


def check_meta_text(prompt: str) -> list[dict[str, Any]]:
    """检查是否把校验报告、修复解释或交付套话混入提示词正文。

    元说明会分散模型注意力并可能被模型当作画面文字。命中项默认作为警告，严格交付模式会阻断并要求把说明
    移到代码块或提示词正文之外。
    """

    issues: list[dict[str, Any]] = []
    for phrase in META_TEXT_PATTERNS:
        for match in re.finditer(re.escape(phrase), prompt):
            issues.append(
                make_issue(
                    prompt,
                    "warning",
                    "meta-text",
                    "提示词疑似混入交付说明或校验过程。",
                    match.start(),
                    match.group(0),
                    "只保留模型需要执行的画面、动作、摄影、声音和必要约束。",
                )
            )
    return issues


def check_delimiters(prompt: str) -> list[dict[str, Any]]:
    """检查 Seedance 声音与文字标签的成对符号是否平衡。

    对每一种标签独立计数并报告首个多余符号；这样既能发现漏闭合，也不会因为不同类型标签交错而产生无关
    的栈顺序误报。
    """

    issues: list[dict[str, Any]] = []
    for opening, closing, label in DELIMITER_PAIRS:
        depth = 0
        for index, character in enumerate(prompt):
            if character == opening:
                depth += 1
            elif character == closing:
                if depth == 0:
                    issues.append(
                        make_issue(
                            prompt,
                            "error",
                            "unbalanced-delimiter",
                            f"{label}标签存在多余的闭合符号“{closing}”。",
                            index,
                            closing,
                            f"补充对应的“{opening}”或删除多余符号。",
                        )
                    )
                else:
                    depth -= 1
        if depth > 0:
            index = prompt.rfind(opening)
            issues.append(
                make_issue(
                    prompt,
                    "error",
                    "unbalanced-delimiter",
                    f"{label}标签缺少 {depth} 个闭合符号“{closing}”。",
                    index,
                    opening,
                    f"补全“{closing}”。",
                )
            )
    return issues


def content_is_negative_or_empty(content: str) -> bool:
    """判断声音或文字字段是否只是“无、不要、关闭”等否定声明。

    Seedance 标签用于要生成的内容；“无对白、无字幕”等禁用要求不需要被标签包裹，也不应被误报为漏标。
    """

    normalized = re.sub(r"[\s，,。.!！?？]", "", content)
    return not normalized or bool(re.match(r"^(?:无|不要|不需要|禁止|关闭|取消|不出现|不生成|没有)", normalized))


def check_seedance_20_labels(prompt: str) -> list[dict[str, Any]]:
    """检查 Seedance 2.0 中显式声音或文字字段是否使用对应标签。

    规则针对“字段名：内容”这种可确定结构，不扫描所有自然语言中的“说、响、写”等词，从而避免把叙事描述
    粗暴改写成标签。负面声明不要求标签；实际生成内容必须由对应括号完整包裹。
    """

    issues: list[dict[str, Any]] = []
    boundary = r"(?:^|[\n。；;])\s*"
    for label, label_pattern, opening, closing in SEEDANCE_20_LABELS:
        pattern = re.compile(boundary + rf"(?P<name>{label_pattern})\s*[:：]\s*(?P<content>[^\n。；;]+)", re.M)
        for match in pattern.finditer(prompt):
            content = match.group("content").strip()
            if content_is_negative_or_empty(content):
                continue
            if content.startswith(opening) and content.endswith(closing):
                continue
            content_index = match.start("content")
            issues.append(
                make_issue(
                    prompt,
                    "error",
                    "untagged-seedance-2.0-content",
                    f"Seedance 2.0 的{label}内容未使用 {opening}…{closing} 标签。",
                    content_index,
                    content[:80],
                    f"将要生成的{label}内容写成 {opening}内容{closing}；若表示不需要，可直接写“无{label}”。",
                )
            )
    return issues


def split_instruction_units(prompt: str) -> list[tuple[str, int]]:
    """把正文切分为可比较的句子或分号级指令单元。

    返回每个单元及其原始起始偏移，供重复检测定位。过短片段通常只是素材编号、时间点或连接词，会在后续
    过滤，避免产生大量低价值相似度告警。
    """

    units: list[tuple[str, int]] = []
    start = 0
    for match in re.finditer(r"[。！？!?；;\n]+", prompt):
        unit = prompt[start : match.start()].strip()
        if unit:
            unit_start = start + len(prompt[start : match.start()]) - len(prompt[start : match.start()].lstrip())
            units.append((unit, unit_start))
        start = match.end()
    tail = prompt[start:].strip()
    if tail:
        tail_start = start + len(prompt[start:]) - len(prompt[start:].lstrip())
        units.append((tail, tail_start))
    return units


def normalize_instruction(text: str) -> str:
    """移除不影响指令含义比较的空白、编号和标点。

    保留汉字、字母、数字及素材引用中的有效字符，使重复检测关注控制内容而非排版差异。该归一化只用于比较，
    不回写提示词。
    """

    text = re.sub(r"^(?:镜头\s*)?\d+\s*[:：、.．-]?\s*", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff@]+", "", text).lower()


def check_duplicate_instructions(prompt: str) -> list[dict[str, Any]]:
    """检查完全重复或高度相似的长指令，防止同一要求争夺注意力。

    完全重复报告错误；长度足够且相似度很高的片段报告警告。比较跳过短片段，并限制只对长度接近的单元计算，
    以减少时间码、素材映射和正常排比造成的误报。
    """

    issues: list[dict[str, Any]] = []
    units = [(raw, index, normalize_instruction(raw)) for raw, index in split_instruction_units(prompt)]
    seen_exact: dict[str, tuple[str, int]] = {}
    comparable: list[tuple[str, int, str]] = []
    for raw, index, normalized in units:
        if len(normalized) < 12:
            continue
        if normalized in seen_exact:
            issues.append(
                make_issue(
                    prompt,
                    "error",
                    "duplicate-instruction",
                    "同一指令在提示词中重复出现。",
                    index,
                    raw[:100],
                    "合并为一次最清楚的表达。",
                )
            )
            continue
        seen_exact[normalized] = (raw, index)
        comparable.append((raw, index, normalized))

    for current_index, (raw, index, normalized) in enumerate(comparable):
        for other_raw, _, other_normalized in comparable[:current_index]:
            length_ratio = min(len(normalized), len(other_normalized)) / max(len(normalized), len(other_normalized))
            if length_ratio < 0.72:
                continue
            similarity = difflib.SequenceMatcher(None, normalized, other_normalized).ratio()
            if similarity < 0.9:
                continue
            issues.append(
                make_issue(
                    prompt,
                    "warning",
                    "similar-instruction",
                    f"两条指令高度相似（相似度 {similarity:.0%}）。",
                    index,
                    raw[:100],
                    f"检查是否可与“{other_raw[:60]}”合并，避免重复控制。",
                )
            )
    return issues


def check_task_binding(prompt: str, task_type: str) -> list[dict[str, Any]]:
    """检查依赖原视频的任务是否明确引用了本次提交素材。

    编辑与延长必须指出具体 ``@视频N``，否则“原视频、结尾、继续”等词仍可能依赖聊天上下文。桥接任务可能
    使用图片关键帧或视频组合，因此这里只要求至少存在一种编号素材引用。
    """

    if task_type in {"edit", "extend"} and not re.search(r"@视频\s*\d+", prompt):
        return [
            make_issue(
                prompt,
                "error",
                "missing-video-binding",
                f"{task_type} 任务没有明确引用本次提交的 @视频N。",
                0,
                prompt[:80],
                "把需要编辑或延长的原视频绑定为 @视频1 等实际编号。",
            )
        ]
    if task_type == "bridge" and not re.search(r"@(?:图片|视频)\s*\d+", prompt):
        return [
            make_issue(
                prompt,
                "error",
                "missing-bridge-binding",
                "桥接任务没有明确引用本次提交的图片或视频素材。",
                0,
                prompt[:80],
                "写明桥接起点与终点所对应的 @图片N 或 @视频N。",
            )
        ]
    return []


def validate_prompt(prompt: str, version: str, task_type: str) -> list[dict[str, Any]]:
    """执行所有可自动判定的通用提示词检查并返回排序后的诊断。

    校验器不尝试替代分镜、素材和空间关系的语义审查；它负责稳定拦截可机械识别的问题。错误优先于警告，
    同级问题按正文位置排序，便于从前到后修复。
    """

    if not prompt.strip():
        return [make_issue(prompt, "error", "empty-prompt", "提示词为空。", 0, "", "提供完整提示词正文。")]

    issues: list[dict[str, Any]] = []
    issues.extend(check_task_binding(prompt, task_type))
    issues.extend(check_external_context(prompt, task_type))
    issues.extend(check_ambiguous_continuations(prompt, task_type))
    issues.extend(check_placeholders(prompt))
    issues.extend(check_meta_text(prompt))
    issues.extend(check_delimiters(prompt))
    issues.extend(check_duplicate_instructions(prompt))
    if version == "2.0":
        issues.extend(check_seedance_20_labels(prompt))

    severity_order = {"error": 0, "warning": 1}
    issues.sort(key=lambda issue: (severity_order[issue["severity"]], issue["line"], issue["column"], issue["code"]))
    return issues


def render_text_result(issues: list[dict[str, Any]], version: str, task_type: str, strict: bool) -> str:
    """把诊断渲染为适合人工阅读的命令行文本。

    输出首先给出版本、任务类型和最终结论，再逐条展示位置、触发片段和修复建议。严格模式会在结论中明确
    说明警告是否阻断，防止调用者把“有警告”误读成已通过门禁。
    """

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    blocked = errors > 0 or (strict and warnings > 0)
    mode = "严格" if strict else "普通"
    lines = [f"Seedance {version} | task={task_type} | mode={mode}"]
    lines.append(f"结果：{'未通过' if blocked else '通过'}（错误 {errors}，警告 {warnings}）")
    for issue in issues:
        label = "错误" if issue["severity"] == "error" else "警告"
        lines.append(f"- [{label}] {issue['code']} @ {issue['line']}:{issue['column']}：{issue['message']}")
        if issue["excerpt"]:
            lines.append(f"  触发：{issue['excerpt']}")
        if issue["suggestion"]:
            lines.append(f"  建议：{issue['suggestion']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """运行命令行校验并返回可供自动化门禁使用的退出码。

    退出码 0 表示当前模式通过，1 表示提示词存在阻断问题，2 表示输入文件或命令本身无法处理。只有脚本实际
    返回 0 时，调用方才能声称可执行门禁已经通过。
    """

    args = parse_args(argv)
    try:
        prompt = read_prompt(args.path)
    except (OSError, UnicodeError) as error:
        if args.json_output:
            print(json.dumps({"passed": False, "input_error": str(error)}, ensure_ascii=False, indent=2))
        else:
            print(f"输入错误：{error}", file=sys.stderr)
        return 2

    task_type = infer_task_type(prompt, args.task)
    issues = validate_prompt(prompt, args.version, task_type)
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    blocked = errors > 0 or (args.strict and warnings > 0)

    if args.json_output:
        print(
            json.dumps(
                {
                    "passed": not blocked,
                    "version": args.version,
                    "task": task_type,
                    "strict": args.strict,
                    "error_count": errors,
                    "warning_count": warnings,
                    "issues": issues,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_text_result(issues, args.version, task_type, args.strict))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())

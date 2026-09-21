#!/usr/bin/env python3
"""测试 Seedance 提示词可执行校验器的通用行为。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = Path(__file__).with_name("validate_prompt.py")


def load_validator() -> ModuleType:
    """从同目录加载校验器模块，避免测试依赖 Skill 目录成为 Python 包。

    Skill 的 scripts 目录不需要额外 ``__init__.py``；测试通过文件路径导入真实交付脚本，确保验证的正是用户
    会执行的实现，而不是复制出的测试替身。
    """

    spec = importlib.util.spec_from_file_location("seedance_validate_prompt", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载校验器：{SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


class ValidatePromptTests(unittest.TestCase):
    """覆盖独立执行、素材续接、标签、重复和占位符等交付风险。"""

    def issue_codes(self, prompt: str, version: str = "2.0", task: str = "new") -> set[str]:
        """返回指定提示词的诊断码集合，方便测试关注规则行为而非完整文案。

        测试显式传入任务类型，避免任务推断变化掩盖真正要验证的规则；版本默认 2.0，以覆盖声音标签门禁。
        """

        return {issue["code"] for issue in VALIDATOR.validate_prompt(prompt, version, task)}

    def test_previous_shot_dependency_fails(self) -> None:
        """上一镜依赖必须被错误级规则拦截。

        这是校验器要解决的核心失效模式：即使句子描述了人物坐姿，“承接上一镜”仍要求模型拥有未提交的上下文。
        """

        prompt = "忧怜承接上一镜，仍坐在妆台前。镜头固定拍摄她转身看向后方。"
        self.assertIn("external-context", self.issue_codes(prompt))

    def test_explicit_opening_state_passes(self) -> None:
        """完整写出开场位置、接触关系和朝向时不应产生续接告警。

        该用例证明校验器鼓励自包含描述，而不是禁止“仍”之外的正常人物动作或固定机位写法。
        """

        prompt = (
            "镜头开始时，忧怜已经坐在妆台前，身体正对妆镜，双手扣住木质台沿，肩背僵硬，头部尚未转向身后。"
            "她突然转头看向后方衣架区域，呼吸急促，镜头保持固定。"
        )
        self.assertEqual(set(), self.issue_codes(prompt))

    def test_extend_bound_to_video_allows_continuation(self) -> None:
        """延长任务就近绑定已提交视频后，可以引用素材结尾并继续动作。

        规则必须区分“模型拿得到的 @视频1”和“模型拿不到的上一镜聊天信息”，否则会误伤官方支持的延长任务。
        """

        prompt = "延长@视频1，从@视频1结尾继续人物向前行走的动作，速度与原视频末尾一致。"
        codes = self.issue_codes(prompt, task="extend")
        self.assertNotIn("external-context", codes)
        self.assertNotIn("ambiguous-continuation", codes)
        self.assertNotIn("missing-video-binding", codes)

    def test_local_action_chain_allows_continuation(self) -> None:
        """正文内已经给出前因的恢复或重复动作不应产生续接警告。

        校验器要阻断的是聊天外依赖，而不是“灯光先熄灭，随后恢复”这类在同一提示词中闭合的动作链。
        """

        prompt = "灯光先短暂熄灭，片刻后恢复正常亮度，人物抬头看向灯具。"
        self.assertNotIn("ambiguous-continuation", self.issue_codes(prompt))

    def test_edit_requires_video_reference(self) -> None:
        """显式编辑任务缺少 @视频N 时必须失败。

        这能防止“修改原视频”等表述依赖聊天中没有随提示词提交的文件身份。
        """

        prompt = "修改原视频中的人物服装颜色，其他内容保持不变。"
        self.assertIn("missing-video-binding", self.issue_codes(prompt, task="edit"))

    def test_edit_bound_to_video_passes(self) -> None:
        """编辑任务明确绑定 @视频N 时不应被当成聊天外依赖。

        该用例与延长任务形成对照，确保通用独立执行规则不会阻断模型实际收到的原视频引用。
        """

        prompt = "编辑@视频1，把人物外套改为深灰色，人物身份、动作、场景、机位和声音保持@视频1原样。"
        codes = self.issue_codes(prompt, task="edit")
        self.assertNotIn("external-context", codes)
        self.assertNotIn("missing-video-binding", codes)

    def test_seedance_20_untagged_sound_fails(self) -> None:
        """2.0 的显式音效字段没有尖括号时必须失败。

        只检查明确的“音效：内容”结构，避免把普通叙事中的声音动词全部误判为标签遗漏。
        """

        prompt = "镜头固定拍摄人物推门进入。音效：木门吱呀声。"
        self.assertIn("untagged-seedance-2.0-content", self.issue_codes(prompt))

    def test_seedance_20_tagged_sound_passes(self) -> None:
        """2.0 使用正确标签且符号平衡时不应报告声音问题。

        同时覆盖持续声场、动作音效、对白和画面文字四类官方写法。
        """

        prompt = "镜头固定拍摄人物推门进入。（远处风声）<木门吱呀声>{人物：谁在那里？}【三更】"
        codes = self.issue_codes(prompt)
        self.assertNotIn("untagged-seedance-2.0-content", codes)
        self.assertNotIn("unbalanced-delimiter", codes)

    def test_seedance_25_does_not_apply_20_label_rule(self) -> None:
        """2.5 文本不应被 2.0 的字段标签规则误伤。

        版本路由必须保持隔离；校验器只能执行所选版本明确启用的机械规则。
        """

        prompt = "镜头固定拍摄人物推门进入。音效：木门吱呀声。"
        self.assertNotIn("untagged-seedance-2.0-content", self.issue_codes(prompt, version="2.5"))

    def test_unbalanced_delimiter_fails(self) -> None:
        """缺少闭合符号的声音标签必须失败。

        成对符号是可确定的语法错误，不能降级为仅供参考的警告。
        """

        prompt = "镜头固定拍摄人物推门进入。<木门吱呀声"
        self.assertIn("unbalanced-delimiter", self.issue_codes(prompt))

    def test_duplicate_instruction_fails(self) -> None:
        """完全重复的长指令必须被检测。

        重复要求会提高非核心信息权重，且常见于多轮修复时把旧补丁继续保留在正文中。
        """

        prompt = "镜头固定在人物肩后拍摄，人物始终位于画面左侧。镜头固定在人物肩后拍摄，人物始终位于画面左侧。"
        self.assertIn("duplicate-instruction", self.issue_codes(prompt))

    def test_placeholder_fails(self) -> None:
        """未替换的素材假编号和模板变量必须失败。

        真实编号可直接提交，字母占位符则意味着素材职责尚未落实。
        """

        prompt = "@图片N用于人物外观，{{场景描述}}，人物从门口走向室内。"
        self.assertIn("unfinished-placeholder", self.issue_codes(prompt))


if __name__ == "__main__":
    unittest.main()

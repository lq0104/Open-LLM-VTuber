from typing import Dict, List, Set, Any, Optional, Callable, Tuple
import os
import yaml
import logging
import time
import re
from pydantic import BaseModel, Field
import difflib
from loguru import logger

# logger = logging.getLogger(__name__)

class StoryChoice(BaseModel):
    choice_id: str
    text: str  # 选项文本
    next_scene: str  # 下一个场景ID
    conditions: Dict[str, Any] = Field(default_factory=dict)  # 选项触发条件
    effects: Dict[str, Any] = Field(default_factory=dict)  # 选择后的效果
    keywords: List[str] = Field(default_factory=list)  # 与选项相关的关键词，用于NLP匹配
    descriptions: List[str] = Field(default_factory=list)  # 描述这个选择的多种说法，用于NLP匹配

class StoryScene(BaseModel):
    scene_id: str
    initial_dialogue: str  # 场景初始对话
    choices: List[StoryChoice] = Field(default_factory=list)  # 可选的对话分支
    conditions: Dict[str, Any] = Field(default_factory=dict)  # 场景触发条件
    achievements: List[str] = Field(default_factory=list)  # 可获得的成就
    is_end_scene: bool = False  # 是否为结束场景
    background: Optional[str] = None  # 场景背景图像路径

class GameState(BaseModel):
    current_scene_id: str
    visited_scenes: Set[str] = Field(default_factory=set)
    achievements: Set[str] = Field(default_factory=set)
    variables: Dict[str, Any] = Field(default_factory=dict)  # 游戏变量，用于条件判断
    dialogue_history: List[Dict] = Field(default_factory=list)  # 对话历史

class DialogueMessage(BaseModel):
    speaker: str  # "character" 或 "user"
    text: str
    timestamp: float

class StoryData(BaseModel):
    title: str
    author: str = "Anonymous"
    description: str = ""
    scenes: Dict[str, StoryScene]
    start_scene: str
    system_prompt: str = ""  # 系统提示词，用于指导大模型的行为
    
class GameManager:
    def __init__(self, service_context):
        self.service_context = service_context
        self.story_data: Optional[StoryData] = None
        self.game_state: Optional[GameState] = None
        self.stories_dir = "stories"
        self.available_stories = self._get_available_stories()
        self.nlp_engine = None  # 使用服务上下文中的大模型
        self.intent_threshold = 0.6  # 意图匹配阈值，低于此值需要确认
        self.similarity_cache = {}  # 缓存相似度计算结果
        
    def _get_available_stories(self) -> Dict[str, str]:
        """获取可用的故事列表"""
        stories = {}
        if not os.path.exists(self.stories_dir):
            os.makedirs(self.stories_dir)
            return stories
            
        for filename in os.listdir(self.stories_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                try:
                    with open(os.path.join(self.stories_dir, filename), 'r', encoding='utf-8') as f:
                        story_data = yaml.safe_load(f)
                        if 'title' in story_data:
                            stories[filename] = story_data['title']
                        else:
                            stories[filename] = filename.replace('.yaml', '').replace('.yml', '').replace('_', ' ').title()
                except Exception as e:
                    logger.error(f"Error loading story {filename}: {e}")
        return stories
        
    def load_story(self, story_filename: str) -> bool:
        """加载指定的故事文件"""
        try:
            filepath = os.path.join(self.stories_dir, story_filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                story_data_dict = yaml.safe_load(f)
                
            # 处理场景数据
            scenes = {}
            for scene_id, scene_data in story_data_dict.get('scenes', {}).items():
                logger.info(f"Processing scene {scene_id}")
                choices = []
                for choice_data in scene_data.get('choices', []):
                    # 处理关键词和描述，用于NLP匹配
                    keywords = choice_data.get('keywords', [])
                    if not keywords and 'text' in choice_data:
                        # 如果没有提供关键词，从选项文本中提取
                        keywords = self._extract_keywords(choice_data['text'])
                        
                    descriptions = choice_data.get('descriptions', [])
                    if not descriptions and 'text' in choice_data:
                        # 如果没有提供描述，使用选项文本作为默认描述
                        descriptions = [choice_data['text']]
                        
                    choices.append(StoryChoice(
                        choice_id=choice_data['choice_id'],
                        text=choice_data['text'],
                        next_scene=choice_data.get('next_scene', ''),
                        conditions=choice_data.get('conditions', {}),
                        effects=choice_data.get('effects', {}),
                        keywords=keywords,
                        descriptions=descriptions
                    ))
                
                scenes[scene_id] = StoryScene(
                    scene_id=scene_id,
                    initial_dialogue=scene_data['initial_dialogue'],
                    choices=choices,
                    conditions=scene_data.get('conditions', {}),
                    achievements=scene_data.get('achievements', []),
                    is_end_scene=scene_data.get('is_end_scene', False),
                    background=scene_data.get('background', None)
                )
            
            self.story_data = StoryData(
                title=story_data_dict.get('title', story_filename.replace('.yaml', '')),
                author=story_data_dict.get('author', 'Anonymous'),
                description=story_data_dict.get('description', ''),
                scenes=scenes,
                start_scene=story_data_dict.get('start_scene', list(scenes.keys())[0]),
                system_prompt=story_data_dict.get('system_prompt', '')  # 加载系统提示词
            )
            return True
        except Exception as e:
            logger.error(f"Error loading story file {story_filename}: {e}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单实现：分词并过滤停用词 
        # TODO: 待优化
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'的', '了', '是', '在', '我', '你', '他', '她', '它', '们', '和', '与', '或', '吗', '啊', '呢', '吧'}
        return [word for word in words if word not in stop_words and len(word) > 1]
        
    def start_game(self, story_filename: str = None) -> Optional[Dict]:
        """开始游戏，初始化游戏状态"""
        # TODO: 游戏开始后，应播放故事的初始对话
        if story_filename and not self.load_story(story_filename):
            return None
            
        if not self.story_data:
            return None
            
        # 初始化游戏状态
        self.game_state = GameState(
            current_scene_id=self.story_data.start_scene,
            visited_scenes={self.story_data.start_scene},
            achievements=set(),
            variables={},
            dialogue_history=[]
        )
        
        # 返回初始场景信息
        return self.get_current_scene_data()
        
    def get_current_scene_data(self) -> Dict:
        """获取当前场景信息，格式化为前端需要的数据"""
        if not self.game_state or not self.story_data:
            return {"error": "Game not initialized"}
            
        scene_id = self.game_state.current_scene_id
        if scene_id not in self.story_data.scenes:
            return {"error": f"Scene {scene_id} not found"}
            
        scene = self.story_data.scenes[scene_id]
        
        # 过滤掉不符合条件的选项
        available_choices = []
        for choice in scene.choices:
            if self._check_conditions(choice.conditions):
                available_choices.append({
                    "choice_id": choice.choice_id,
                    "text": choice.text
                })
        
        response_data = {
            "scene_id": scene.scene_id,
            "dialogue": scene.initial_dialogue,
            "choices": available_choices,
            "is_end_scene": scene.is_end_scene
        }
        
        if scene.background and os.path.exists(scene.background):
            response_data["background"] = os.path.basename(scene.background)
            
        return response_data
        
    def _check_conditions(self, conditions: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        # TODO: 如果条件没有达到，玩家能知道什么原因没达到吗，要不要给玩家一些提示
        if not conditions:
            return True
            
        for var_name, required_value in conditions.items():
            if var_name == "visited_scenes":
                # 特殊处理：检查场景是否已访问
                if isinstance(required_value, list):
                    for scene in required_value:
                        if scene not in self.game_state.visited_scenes:
                            return False
                else:
                    if required_value not in self.game_state.visited_scenes:
                        return False
            elif var_name == "achievements":
                # 特殊处理：检查成就
                if isinstance(required_value, list):
                    for achievement in required_value:
                        if achievement not in self.game_state.achievements:
                            return False
                else:
                    if required_value not in self.game_state.achievements:
                        return False
            else:
                # 检查变量
                if var_name not in self.game_state.variables:
                    return False
                if self.game_state.variables[var_name] != required_value:
                    return False
        return True
        
    def _apply_effects(self, effects: Dict[str, Any]):
        """应用效果到游戏状态"""
        if not effects:
            return
            
        for var_name, value in effects.items():
            if var_name == "add_achievement":
                if isinstance(value, list):
                    for achievement in value:
                        self.game_state.achievements.add(achievement)
                else:
                    self.game_state.achievements.add(value)
            else:
                # 设置变量
                self.game_state.variables[var_name] = value
                
    def process_user_choice(self, choice_id: str) -> Dict:
        """处理用户直接选择，更新游戏状态"""
        if not self.game_state or not self.story_data:
            return {"error": "Game not initialized"}
            
        current_scene = self.story_data.scenes.get(self.game_state.current_scene_id)
        if not current_scene:
            return {"error": "Current scene not found"}
            
        # 查找选择
        next_scene_id = None
        for choice in current_scene.choices:
            if choice.choice_id == choice_id and self._check_conditions(choice.conditions):
                # TODO: 如果条件没有达到，玩家能知道什么原因没达到吗，要不要给玩家一些提示
                # 应用效果
                self._apply_effects(choice.effects)
                next_scene_id = choice.next_scene
                
                # 添加对话到历史
                self.game_state.dialogue_history.append({
                    "speaker": "user",
                    "text": choice.text,
                    "timestamp": time.time()
                })
                break
                
        if not next_scene_id:
            return {"error": f"Choice {choice_id} not found or invalid"}
            
        # 更新当前场景
        self.game_state.current_scene_id = next_scene_id
        self.game_state.visited_scenes.add(next_scene_id)
        
        # 如果新场景有成就，添加到玩家成就中
        new_scene = self.story_data.scenes.get(next_scene_id)
        if new_scene and new_scene.achievements:
            for achievement in new_scene.achievements:
                self.game_state.achievements.add(achievement)
        
        # 添加角色对话到历史
        if new_scene:
            self.game_state.dialogue_history.append({
                "speaker": "character",
                "text": new_scene.initial_dialogue,
                "timestamp": time.time()
            })
                
        # 返回新场景信息
        return self.get_current_scene_data()

    async def process_agent_response(self, input_text: str) -> Dict:
        """处理用户自由输入文本，使用大模型判断用户意图"""
        if not self.game_state or not self.story_data:
            return {"error": "Game not initialized"}
            
        # 添加对话到历史
        self.game_state.dialogue_history.append({
            "speaker": "user",
            "text": input_text,
            "timestamp": time.time()
        })
        
        current_scene = self.story_data.scenes.get(self.game_state.current_scene_id)
        if not current_scene:
            return {"error": "Current scene not found"}
        
        # 过滤出符合条件的选项
        available_choices = [
            choice for choice in current_scene.choices 
            if self._check_conditions(choice.conditions)
        ]
        
        if not available_choices:
            # 没有可用选项，返回错误
            return {"error": "No available choices in current scene"}
        
        # 使用大模型匹配用户意图
        match_result = await self._match_intent_with_llm(input_text, current_scene.initial_dialogue, available_choices)
        
        if match_result.confidence >= self.intent_threshold:
            # 匹配度足够高，直接处理选择
            choice_result = self.process_user_choice(match_result.choice_id)
            # 添加语言帮助信息到返回结果
            if hasattr(match_result, 'language_help') and match_result.language_help:
                choice_result['language_help'] = match_result.language_help
            return choice_result
        else:
            # 匹配度不够，使用大模型生成的对话回应
            response_data = self.get_current_scene_data()
            response_data["dialogue"] = match_result.dialogue_response
            # 添加语言帮助信息到返回结果
            if hasattr(match_result, 'language_help') and match_result.language_help:
                response_data['language_help'] = match_result.language_help
            return response_data
    
    class MatchResult:
        def __init__(self, choice_id: str, confidence: float):
            self.choice_id = choice_id
            self.confidence = confidence
            self.dialogue_response = ""  # 添加对话回应字段
            self.language_help = {}  # 添加语言帮助字段
    
    async def _match_intent_with_llm(self, input_text: str, initial_dialogue: str, choices: List[StoryChoice]) -> MatchResult:
        """使用大模型匹配用户意图"""
        if not choices:
            return self.MatchResult("", 0.0)
            
        # 构建提示词
        prompt = f"""
当前场景的初始对话是：
{initial_dialogue}

当前场景有以下选项：
{chr(10).join([f"{i+1}. {choice.text}" for i, choice in enumerate(choices)])}

玩家的输入是: "{input_text}"

请分析玩家的输入最可能对应哪个选项。考虑玩家输入的语义、意图和关键词，而不仅仅是字面匹配。

注意事项：
1. 玩家是学习英语的中国人
2. 只有在以下情况下才提供语言帮助：
   a) 用户明确询问某个表达如何用英语说（例如："苹果怎么说"）
   b) 用户使用了明显错误的英语表达
   c) 用户明确表示听不懂某个英语表达
3. 如果玩家的英语表达不准确但可以理解，不要纠正
4. 如果玩家可能听不懂某些英语表达，请使用更简单的英语或中文回应
5. 请不要在对话回应中包含你的内部推理过程或决策说明

如果玩家输入与某个选项的匹配度很高，只需输出该选项的编号（1到{len(choices)}之间的整数）。
如果玩家输入与任何选项的匹配度都不高，请生成一个自然的对话回应，继续当前的话题。

输出格式：
场景选项: [选项编号，必须是1到{len(choices)}之间的整数，如果匹配度高]
对话回应: [如果匹配度低，生成一个自然的对话回应，不要包含你的内部思考]
语言帮助: [仅在满足条件2时提供，格式为{{\"用户不知道怎么表达的中文\": \"对应的英文\", \"用户错误的英文表达\": \"正确的英文表达\"}}]
"""
        
        try:
            # 使用自定义的方式向LLM模型发送请求
            from ..agent.input_types import TextData, TextSource, BatchInput
            
            # 创建输入对象
            input_data = BatchInput(
                texts=[TextData(source=TextSource.INPUT, content=prompt)]
            )
            
            # 调用LLM模型
            agent_engine = self.service_context.agent_engine
            response_text = ""
            
            start_time = time.time()
            # 收集LLM输出的所有token
            async for response in agent_engine.chat(input_data):
                if hasattr(response, 'tts_text'):
                    response_text += response.tts_text
            end_time = time.time()
            logger.info(f"🔊 LLM intent matching response took {end_time - start_time} seconds")
            logger.info(f"LLM intent matching response: {response_text}")
            
            # 解析回复，提取选项编号和对话回应
            # 使用更灵活的正则表达式来匹配可能的格式
            option_match = re.search(r'场景选项:?\s*(\d+)', response_text)
            dialogue_match = re.search(r'对话回应:?[ \t]*(.*?)(?=\s*语言帮助:|\n|$)', response_text, re.DOTALL)
            language_help_match = re.search(r'语言帮助:?[ \t]*(\{.*?\}|\[.*?\]|.+?)(?=\s*对话回应:|\n|$)', response_text, re.DOTALL)
            
            option_index = -1
            dialogue_response = ""
            language_help = {}
            
            if option_match:
                option_index = int(option_match.group(1))
                logger.info(f"Matched option index: {option_index}")
                
            if dialogue_match:
                dialogue_response = dialogue_match.group(1).strip()
                logger.info(f"Matched dialogue response: {dialogue_response}")
                
            if language_help_match:
                language_help_text = language_help_match.group(1).strip()
                logger.info(f"Matched language help: {language_help_text}")
                try:
                    # 尝试提取语言帮助信息，可能的格式不同
                    if language_help_text and language_help_text != "[]" and language_help_text != "{}":
                        # 尝试解析为JSON
                        import json
                        try:
                            language_help = json.loads(language_help_text)
                        except:
                            # 如果JSON解析失败，尝试简单提取
                            language_help = {"提取的语言帮助": language_help_text}
                except Exception as e:
                    logger.error(f"Error parsing language help: {e}")
            
            # 构造并发送调试信息
            debug_info = {
                "llm_response": response_text,
                "intent_match": {
                    "option_index": option_index,
                    "dialogue_response": dialogue_response,
                    "language_help": language_help
                }
            }
            
            # 如果成功解析了选项
            if option_match:
                option_index = int(option_match.group(1)) - 1  # 转换为0-based索引
                
                # 检查选项索引是否有效
                if 0 <= option_index < len(choices):
                    # 如果匹配到选项，设置较高的置信度
                    result = self.MatchResult(choices[option_index].choice_id, 0.8)
                    result.dialogue_response = dialogue_response
                    result.language_help = language_help
                    return result
                else:
                    logger.warning(f"Invalid option index: {option_index}, choices length: {len(choices)}")
            
            # 如果解析失败或选项索引无效，使用备用方法
            logger.info("Using fallback method for intent matching")
            fallback_result = self._match_intent_fallback(input_text, choices)
            # 只有在没有语言帮助时才使用默认对话响应
            if not language_help:
                fallback_result.dialogue_response = dialogue_response or "我理解你想继续对话。让我们继续当前的话题吧。"
            else:
                # 如果有语言帮助，使用对话响应或留空
                fallback_result.dialogue_response = dialogue_response or ""
            fallback_result.language_help = language_help
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error in LLM intent matching: {e}")
            # 错误时使用备用方法
            fallback_result = self._match_intent_fallback(input_text, choices)
            # 只在没有语言帮助时显示默认回复
            if not language_help:
                fallback_result.dialogue_response = "我理解你想继续对话。让我们继续当前的话题吧。"
            fallback_result.language_help = language_help
            return fallback_result
    
    def _match_intent_fallback(self, input_text: str, choices: List[StoryChoice]) -> MatchResult:
        """当LLM匹配失败时，使用备用的关键词和相似度匹配"""
        best_match = None
        best_score = 0
        
        input_lower = input_text.lower()
        
        for choice in choices:
            # 1. 检查关键词匹配
            keyword_score = 0
            for keyword in choice.keywords:
                if keyword.lower() in input_lower:
                    keyword_score += 0.2  # 每匹配一个关键词增加0.2分
            
            # 2. 检查描述相似度
            max_similarity = 0
            for description in choice.descriptions:
                similarity = self._get_text_similarity(input_lower, description.lower())
                max_similarity = max(max_similarity, similarity)
            
            # 3. 综合分数
            total_score = 0.3 * keyword_score + 0.7 * max_similarity
            
            if total_score > best_score:
                best_score = total_score
                best_match = choice.choice_id
        
        return self.MatchResult(best_match or choices[0].choice_id, best_score)
    
    def _get_text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        # 使用缓存
        cache_key = f"{text1}|{text2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # TODO: 使用大模型计算相似度
        # 使用difflib计算相似度
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 缓存结果
        self.similarity_cache[cache_key] = similarity
        return similarity
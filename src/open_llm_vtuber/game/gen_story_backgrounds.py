#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import time
from pathlib import Path
from gen_image import generate, save_binary_file
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def load_api_key():
    """从conf.yaml加载配置"""
    with open('conf.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 获取Gemini API密钥
    api_key = config.get('character_config', {}).get('agent_config', {}).get('llm_configs', {}).get('gemini_llm', {}).get('llm_api_key', '')
    if not api_key:
        print("警告：未找到Gemini API密钥，将使用默认密钥")
        exit()
    return api_key

def load_story():
    """从example_story.yaml加载故事内容"""
    with open('stories/example_story.yaml', 'r', encoding='utf-8') as f:
        story = yaml.safe_load(f)
    return story

def create_prompt_for_scene(scene_id, scene_data, story_title):
    """为场景创建图像生成提示"""
    # 从场景描述中提取关键信息
    scene_description = scene_data.get('initial_dialogue', '')
    scene_description = scene_description.replace('\n', ' ')
    # 创建适合图像生成的提示
    prompt = f"为故事《{story_title}》中的场景“{scene_id}”创建一张背景图。"
    # prompt += f"你可以通过以下描述提取一些关于环境的关键字来生成背景图：{scene_description}。"
    prompt += "图像应具有细节感、氛围感，适合作为视觉小说或游戏的背景。"
    prompt += "请使图像在构图和光影上具有视觉吸引力。"
    prompt += "避免添加任何文字或干扰元素，以便背景图能够清晰地传达场景信息。"
    
    return prompt

def main():
    # 加载配置
    api_key = load_api_key()
    
    # 加载故事
    story = load_story()
    story_title = story.get('title', 'Unknown Story')
    
    # 确保backgrounds目录存在
    os.makedirs(f'backgrounds/{story_title}', exist_ok=True)
    
    # 处理每个场景
    scenes = story.get('scenes', {})
    for scene_id, scene_data in scenes.items():
        # 检查是否已有background字段且图像存在
        background_path = scene_data.get('background', '')
        if background_path and os.path.exists(background_path):
            print(f"场景 '{scene_id}' 已有背景图像: {background_path}")
            continue
            
        # 创建图像生成提示
        prompt = create_prompt_for_scene(scene_id, scene_data, story_title)
        
        # 设置输出文件名
        output_file = f"backgrounds/{story_title}/{scene_id}"
        
        print(f"正在为场景 '{scene_id}' 生成背景图像...")
        print(f"提示: {prompt}")
        
        # 调用generate函数生成图像，传递API密钥
        try:
            data_buffer, file_extension, chunk_text = generate(prompt, api_key)
        except Exception as e:
            print(f"生成图像时发生错误: {e}")
            continue
        
        # 更新story中的background字段
        if file_extension:
            print(f"generate image chunk_text: {chunk_text}")
            scene_data['background'] = f"{output_file}{file_extension}"
            save_binary_file(f"{output_file}{file_extension}", data_buffer)
        
        # 由于Gemini API限制为10RPM，添加延迟
        print("等待6秒以遵守API速率限制...")
        time.sleep(6)
    
    # 保存更新后的story到文件
    with open('stories/example_story.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(story, f, allow_unicode=True, sort_keys=False)
    
    print("所有场景的背景图像生成完成！")

if __name__ == "__main__":
    main() 
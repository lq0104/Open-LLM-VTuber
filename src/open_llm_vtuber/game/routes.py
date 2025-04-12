from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, Depends, Query
from starlette.websockets import WebSocketDisconnect
from loguru import logger

from ..service_context import ServiceContext
from .story import GameManager


def init_game_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    创建并返回处理互动小说游戏的API路由。
    
    Args:
        default_context_cache: 默认服务上下文缓存
        
    Returns:
        APIRouter: 配置好的路由器
    """
    
    router = APIRouter(prefix="/game", tags=["game"])
    
    # 创建一个游戏管理器实例
    game_manager = GameManager(default_context_cache)
    
    # 存储每个用户会话的游戏状态
    user_game_sessions: Dict[str, GameManager] = {}
    
    def get_user_game_manager(user_id: str) -> GameManager:
        """获取或创建用户的游戏管理器"""
        if user_id not in user_game_sessions:
            # 为新用户创建一个新的游戏管理器实例
            user_game_sessions[user_id] = GameManager(default_context_cache)
        return user_game_sessions[user_id]
    
    @router.get("/stories")
    async def get_available_stories() -> Dict[str, str]:
        """获取所有可用的故事列表"""
        return game_manager.available_stories
    
    @router.post("/start")
    async def start_game(user_id: str, story_filename: str) -> Dict:
        """
        开始一个新游戏
        
        Args:
            user_id: 用户ID
            story_filename: 故事文件名
            
        Returns:
            Dict: 初始场景数据
        """
        user_game = get_user_game_manager(user_id)
        
        # 开始游戏并获取初始场景
        initial_scene = user_game.start_game(story_filename)
        if initial_scene is None:
            raise HTTPException(status_code=404, detail=f"故事 {story_filename} 加载失败")
            
        return initial_scene
    
    @router.post("/choice")
    async def process_choice(user_id: str, choice_id: str) -> Dict:
        """
        处理用户的选择
        
        Args:
            user_id: 用户ID
            choice_id: 用户选择的选项ID
            
        Returns:
            Dict: 下一个场景数据
        """
        user_game = get_user_game_manager(user_id)
        
        # 处理用户选择
        result = user_game.process_user_choice(choice_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    
    @router.post("/input")
    async def process_user_input(user_id: str, input_text: str) -> Dict:
        """
        处理用户的自由输入文本
        
        Args:
            user_id: 用户ID
            input_text: 用户输入文本
            
        Returns:
            Dict: 处理结果和下一个场景数据
        """
        user_game = get_user_game_manager(user_id)
        
        # 异步处理用户输入
        result = await user_game.process_user_input(input_text)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
        
    @router.get("/current-scene")
    async def get_current_scene(user_id: str) -> Dict:
        """
        获取当前场景数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 当前场景数据
        """
        user_game = get_user_game_manager(user_id)
        
        # 获取当前场景
        result = user_game.get_current_scene_data()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    
    @router.websocket("/ws")
    async def game_websocket(websocket: WebSocket, user_id: str = Query(...)):
        """
        游戏WebSocket连接，支持实时交互
        
        Args:
            websocket: WebSocket连接
            user_id: 用户ID
        """
        await websocket.accept()
        logger.info(f"Game WebSocket connection established for user {user_id}")
        
        try:
            user_game = get_user_game_manager(user_id)
            # 设置WebSocket连接到GameManager
            user_game.set_websocket(websocket)
            
            while True:
                # 接收用户消息
                data = await websocket.receive_json()
                action = data.get("action")
                
                if action == "start":
                    story_filename = data.get("story_filename")
                    if not story_filename:
                        await websocket.send_json({"error": "缺少故事文件名"})
                        continue
                        
                    # 开始游戏
                    initial_scene = user_game.start_game(story_filename)
                    if initial_scene is None:
                        await websocket.send_json({"error": f"故事 {story_filename} 加载失败"})
                    else:
                        await websocket.send_json({"type": "scene", "data": initial_scene})
                        
                elif action == "choice":
                    choice_id = data.get("choice_id")
                    if not choice_id:
                        await websocket.send_json({"error": "缺少选择ID"})
                        continue
                        
                    # 处理用户选择
                    result = user_game.process_user_choice(choice_id)
                    await websocket.send_json({"type": "scene", "data": result})
                    
                elif action == "input":
                    input_text = data.get("text")
                    if not input_text:
                        await websocket.send_json({"error": "缺少输入文本"})
                        continue
                        
                    # 发送调试信息 - 用户输入
                    await user_game.send_debug_info({
                        "user_input": input_text
                    })
                    
                    # 异步处理用户输入
                    result = await user_game.process_user_input(input_text)
                    await websocket.send_json({"type": "scene", "data": result})
                    
                elif action == "get_scene":
                    # 获取当前场景
                    result = user_game.get_current_scene_data()
                    await websocket.send_json({"type": "scene", "data": result})
                    
                else:
                    await websocket.send_json({"error": f"未知操作 {action}"})
                    
        except WebSocketDisconnect:
            logger.info(f"Game WebSocket client {user_id} disconnected")
            # 清理用户游戏管理器的WebSocket连接
            if user_id in user_game_sessions:
                user_game_sessions[user_id].websocket = None
        except Exception as e:
            logger.error(f"Error in game WebSocket connection: {e}")
            # 清理用户游戏管理器的WebSocket连接
            if user_id in user_game_sessions:
                user_game_sessions[user_id].websocket = None
            await websocket.close()
    
    return router 
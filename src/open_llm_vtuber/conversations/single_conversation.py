from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from ..game.story import GameManager
from ..agent.output_types import SentenceOutput, Actions, DisplayText

async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
        )

        # Store user message
        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )
        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        # Process agent response
        full_response = await process_agent_response(
            context=context,
            batch_input=batch_input,
            websocket_send=websocket_send,
            tts_manager=tts_manager,
        )

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)


async def process_agent_response(
    context: ServiceContext,
    batch_input: Any,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
) -> str:
    """Process agent response and generate output

    Args:
        context: Service context containing all configurations and engines
        batch_input: Input data for the agent
        websocket_send: WebSocket send function
        tts_manager: TTSTaskManager for the conversation

    Returns:
        str: The complete response text
    """
    full_response = ""
    try:
        # 检查是否是故事模式
        if hasattr(context, 'game_mode') and context.game_mode == "story":
            # 使用story.py中的process_agent_response函数
            response_data = await context.game_manager.process_agent_response(batch_input.texts[0].content)
            
            # 如果返回的是字典，说明是故事模式的响应
            if isinstance(response_data, dict):
                # 更新当前场景数据
                context.current_scene_data = response_data
                
                # 构建完整的响应文本
                response_text = response_data.get("dialogue", "")
                dialogue_response = SentenceOutput(
                    display_text=DisplayText(text=response_text),
                    tts_text=response_text,
                    actions=Actions()
                )
                full_response = await process_agent_output(
                    output=dialogue_response,
                    character_config=context.character_config,
                    live2d_model=context.live2d_model,
                    tts_engine=context.tts_engine,
                    websocket_send=websocket_send,
                    tts_manager=tts_manager,
                    translate_engine=context.translate_engine,
                )
                
                # # 如果有选项，添加到响应中
                # if "choices" in response_data:
                #     choices_text = "\n".join([f"{i+1}. {choice['text']}" for i, choice in enumerate(response_data["choices"])])
                #     response_text += f"\n\n{choices_text}"
                
                return full_response
        
        # 非故事模式或故事模式处理失败，使用默认处理
        agent_output = context.agent_engine.chat(batch_input)
        async for output in agent_output:
            response_part = await process_agent_output(
                output=output,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
                translate_engine=context.translate_engine,
            )
            full_response += response_part

    except Exception as e:
        logger.error(f"Error processing agent response: {e}")
        raise

    return full_response
